import xml.etree.ElementTree as ET
import pandas as pd
import os
import numpy as np

from fox_toolbox.utils.rates import Curve, RateCurve, Swap, Swaption, Volatility

# parse_1d_curve returns now Curve object

def get_files(regexes, folder=None): 
    files = [file for file in os.listdir(folder) if all(x in file for x in regexes.split())]
    if folder:
        files = [os.path.join(folder,file) for file in files]
    return files[0] if len(files) == 1 else files 

def get_xml(fpath):
    xml_tree = ET.parse(fpath)
    xml_root = xml_tree.getroot()
    return xml_tree, xml_root


def get_str_node(root_node, xpath, default=None):
    val = root_node.findtext(xpath)
    return val.strip() if val is not None else default


def get_float_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    return float(val) if val is not None else default


def get_int_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    return int(val) if val is not None else default


def get_delim_str_node(root_node, xpath, default=None):
    val = get_str_node(root_node, xpath, None)
    if val is None:
        return [default]
    return list(map(str.strip, val.split(';')))


def get_delim_float_node(root_node, xpath, default=None):
    return delim_node_to_array(root_node.find(xpath), default)


def delim_node_to_array(node, default=None):
    return np.array(node.text.split(';'), dtype=float) if node is not None else np.array([default], dtype=float)


def parse_1d_curve(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'OneDCurve':
        raise ValueError("Not a 1d curve node.")

    label = get_str_node(node, 'CurveLabel', '')
    buckets = get_delim_float_node(node, 'Buckets')
    values = get_delim_float_node(node, 'Values')
    interp = get_str_node(node, 'InterpolationMethod')
    return Curve(buckets, values, interp, label)


def parse_2d_curve(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'TwoDCurve':
        raise ValueError("Not a 2d curve node.")

    interp = get_str_node(node, 'InterpolationMethod')
    buckets_x = get_delim_float_node(node, 'Buckets')
    buckets_y = get_delim_float_node(node, 'Values/OneDCurve/Buckets')
    values = np.array([delim_node_to_array(n) for n in node.findall('Values/OneDCurve/Values')])
    return interp, buckets_x, buckets_y, values


def get_rate_curve(curve_node):
    if curve_node is None:
        raise ValueError("Argument is null.")

    if curve_node.tag != 'OneDCurve':
        if curve_node[0].tag == 'OneDCurve':
            curve_node = curve_node[0]
        else:
            raise ValueError("Cannot find curve.")

    oneDcurve = parse_1d_curve(curve_node)
    return RateCurve.from_curve(oneDcurve)


def get_rate_curves(node_or_file):
    "returns main curve and list of spreads curve as rates.RateCurve"
    process_node = _get_node(node_or_file, node_name = 'Process')  
    
    assert process_node.tag in ['Process', 'Query'], f'argument should be <Process> or <Query> node, but not {process_node.tag}'
    
    main_curve = get_rate_curve(process_node.find('.//RateCurve'))
    sprd_curve_nodes = process_node.findall('.//SpreadRateCurveList/SpreadRateCurves/SpreadRateCurve/OneDCurve')
    sprd_curves = []
    for node in sprd_curve_nodes:
        curve = parse_1d_curve(node)
        rate_curve = RateCurve(curve.buckets, curve.values + main_curve.zc_rates, curve.interpolation_mode, curve.label)
        sprd_curves.append(rate_curve)
    if len(sprd_curves) == 0:
        sprd_curves = None
    return main_curve, sprd_curves


def get_hw_params(xmlfile):
    params_node = xmlfile.find('.//FactorsList/Factors/Factor')
    curve = parse_1d_curve(params_node.find('VolatilityCurve/OneDCurve'))
    mr = get_float_node(params_node, 'MeanRR')
    return mr, (curve.buckets, curve.values)
    

def get_calib_instr(node):
    if node is None:
        raise ValueError("Argument is null.")
    if node.tag != 'CalibrationInstrument':
        raise ValueError("Not a calibration instrument.")

    atm_vol = get_float_node(node, 'BlackVol')
    if atm_vol == 0.0:
        atm_vol = get_float_node(node, 'SmileSlice/OneDCurve/Values')
    
    vol_nature = node.find('VolModel')
    if vol_nature is None:
        shift = get_float_node(node, 'SmileShift', 0.0)
    else:
        shift = None
    vol = Volatility(atm_vol, 'SLN' if vol_nature is None else 'N', shift)
    flows = [f.text.split(';') for f in node.findall('Flows/Flow')]
    fixing_date = float(flows[0][0])
    pmnt_dates = np.array([f[0] for f in flows[1:-1]], dtype=float)
    dcfs = np.array([f[2] for f in flows[1:-1]], dtype=float)

    return Swaption(
        get_float_node(node, 'OptionExpiry'),
        vol,
        fixing_date,
        pmnt_dates,
        dcfs,
        get_int_node(node, 'CalInstTenor'),
        lvl=get_int_node(node, 'CalibrationLevel'),
        pay_rec = get_str_node(node, 'PayReceive'),
        cal_type=get_str_node(node, 'CalInstKType'),
        strike=get_float_node(node, 'Strike', 0.) / 100., #if no strike in irsm log => strike == 0
        cal_vol=get_float_node(node, 'CalibratedVolatility', np.nan),
        fwd=get_float_node(node, 'SwapRateInfo/SpotSwapRate', np.nan),
        annuity=get_float_node(node, 'SwapRateInfo/FixLeg', np.nan),
        tgt_premium=get_float_node(node, 'BlackPrice', np.nan),
        cal_premium=get_float_node(node, 'CalibratedPremium', np.nan),
        fwdAdj=get_float_node(node, 'SwapRateInfo/FwdSwapRateCmsRep', np.nan),
        fwdAdjModel=get_float_node(node, 'SwapRateInfo/FwdSwapRateModel', np.nan),
    )


def get_calib_basket(xmlfile):
    if isinstance(xmlfile, str):
        _, xmlfile = get_xml(xmlfile)
    for instr in xmlfile.iterfind(".//CalibrationInstrument"):
        yield get_calib_instr(instr)


def parse_process(process_node):
    assert process_node.tag == 'Process', 'should be <Process> node'
    main_curve, spread_curves = get_rate_curves(process_node)
    dic_ = {
        'pillars': main_curve.curve_pillars,
        main_curve.label: main_curve.zc_rates

    }
    for spread_curve in spread_curves:
        dic_[spread_curve.label] = spread_curve.zc_rates
    return pd.DataFrame(dic_)


def parse_process_list(xml_node):
    """xml_node = BucketProcess"""
    dfs = []
    for p in xml_node.findall('.//Process'):
        if p.find('RateCurve') is None:
            continue
        df = parse_process(p)
        df['ccy'] = get_str_node(p, './/ProcessLabel')
        yield df


def parse_debug(xml_node):
    # Nikita
    if isinstance(xml_node, str):
        _, xml_node = get_xml(xml_node)
        
    qfields = 'QRType QRLabel QRAsset QRLevel QRRefCurveLabel'.split(' ')
    dfs = []
    for query in xml_node.findall('.//Query'):
        qdfs = []
        qr_buckets_list = get_delim_float_node(query, './/QRBuckets/OneDCurve/Buckets', 0.0)
        for qr_bucket, process_list in zip(qr_buckets_list, query.findall('.//BucketProcess')):
            df_gen = parse_process_list(process_list)
            df = pd.concat(list(df_gen))
            df['Bucket'] = qr_bucket
            qdfs.append(df)

        qdf = pd.concat(qdfs, ignore_index=True)  # concat by bump pillars
        for qfield in qfields:
            qdf[qfield] = get_str_node(query, f'.//{qfield}')
        dfs.append(qdf)
    return pd.concat(dfs, ignore_index=True)  # concat by queries


def _get_bumped_curves(xmlfile):
    main_curve, sprds = get_rate_curves(xmlfile.find('.//Process'))
    labels = [curve.label for curve in sprds]
    labels.insert(0, main_curve.label)
    df = parse_debug(xmlfile)
    for lbl in labels:
        yield df.pivot_table(values=[lbl], index=['ccy','pillars'], columns=['QRType', 'Bucket'])


def get_bumped_curves(xml_file):
    """yield data frame with bump scenarios for some curve label (ex. USD STD)"""
    _, xml_tree = get_xml(xml_file)
    return _get_bumped_curves(xml_tree)

def parse_rate_curves(xml_file):
    _, xml_tree = get_xml(xml_file)
    return parse_process_list(xml_tree)

def get_curve_bumps(xml_name):
    #  Nikita: not yet implemented
    bumped_curves = get_bumped_curves(xml_name)
    for crv in bumped_curves:
        yield crv 

        
def _get_node(node_or_file, node_name):
    'parse node or file to return <Process> node'
    
    regex = f'.//{node_name}'
    
    if isinstance(node_or_file, str):
        _, tree = get_xml(node_or_file)
        return tree.find(regex)
    elif ET.iselement(node_or_file):
        if not node_or_file.tag == node_name:
            return node_or_file.find(regex)
        else:
            return node_or_file
    else:
        TypeError(f'input should be str or ET.tree not {type(node_or_file)}')