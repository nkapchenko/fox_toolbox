from macs_service.date_util import generate_date, generate_dates

"""-----------------------------------------SERVICE MARKET DATA EXAMPLE----------------------------------------------
http://ci-macsi/jenkins/job/service_pipelines/job/dev/External_20docs/"""


curve = {
        "dates": ["2000-12-05T00:00:00.000Z", "2030-12-05T00:00:00.000Z"],
        "zeroRates": [0.01, 0.01]
        }

smiles = {
           'optionExpiries': ['1Y'],
           'strikes': [-1., 0., 0.5, 1.],
           'swapTenors': ['1Y', '3Y'],
           'volatilities': [0.03] * 2 * 4,  # #tenor x #strikes
           'volatilityType': 'NORMAL'
          }

correls = {
            'correlations': [0.5],
            'dates': ['1Y'],
            'swapTenors': ['1Y', '3Y']
            }

"""-----------------------------------------GET MARKET DATA-------------------------------------------------"""



def _get_curve_mkt_data(ccy, curve):
    return {'rates': {
                        ccy: {'rateCurve': curve}
                    }
            }

def add_swo_vol_mkt_data(swap_mkt, ccy, smiles):
    swap_mkt['rates'][ccy]['horizontalSwaptionVolatilities'] = smiles

def _get_swo_vol_mkt_data(ccy, curve, smiles):
    swo_mkt = _get_curve_mkt_data(ccy, curve)
    add_swo_vol_mkt_data(swo_mkt, ccy, smiles)
    return swo_mkt

def add_swo_correl_mkt_data(swo_mkt, ccy, expiry, correl, tenors):
    swo_mkt['rates'][ccy]['swapTerminalCorrelations'] = {
                                            'correlations': [correl],
                                            'dates': [expiry],
                                            'swapTenors': list(tenors)
                                            }


def _get_swo_correl_mkt_data(ccy, curve, smiles, expiry, correl, tenors):
    swo_correl_mkt = _get_swo_vol_mkt_data(ccy, curve, smiles)
    add_swo_correl_mkt_data(swo_correl_mkt, ccy, expiry, correl, tenors)
    return swo_correl_mkt


"""----------------------------------------GET PAYOFF----------------------------------------------------"""


def _get_swap_simple(self):
    return  {
                 "nominal": self.nominal,
                 "startDate": self.start,
                 "fixedRate": self.strike,
                 "tenor": self.tenor,
                 "currency": self.ccy,
                "fixedRateDirection": "PAYER" if self.is_payer else "RECEIVER"
              }



def _get_swap(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N):
    assert start <= end, 'start > end'

    # might be simplified with itertools.product([True, False], [floatFreq, fixFreq])
    startDatesRec, endDatesRec = [generate_dates(start, end, floatFreq, includeStart) for includeStart in [True, False]]
    startDatesPay, endDatesPay = [generate_dates(start, end, fixFreq, includeStart) for includeStart in [True, False]]


    return {
        "nominal": N,
        "payerLeg": {
            "fixedRate": fixRate,
            "schedule": {
                "startDates": startDatesPay,
                "endDates": endDatesPay,
                "paymentDates": endDatesPay,
                "dayCountConvention": "30/360"
            },
            "paymentCurrency": ccy
        },
        "receiverLeg": {
            "schedule": {
                "fixingDates": startDatesRec,
                "startDates": startDatesRec,
                "endDates": endDatesRec,
                "paymentDates": endDatesRec,
                "dayCountConvention": "ACT/365"
            },
            "rateIndex": f"{ccy}LIBOR" + floatFreq,
            "rateSpread": spread,
            "paymentCurrency": ccy
        },
    }


def _get_swo(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N, expiry):
    return {
                "swap": _get_swap(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N),
                "exerciseDate": expiry
                }


"""-----------------------------------------------GET TASK--------------------------------------------------"""

def _get_task_form(ccy, curve, asof):
    return {
        "marketData": _get_curve_mkt_data(ccy, curve),
        "settings": {
                    "pricingDate": generate_date(asof),
                    "hardware": "CPU",
                    "numericalMethod": "INTRINSIC",
                    "modelPrecision": "SECOND_ORDER"
                },
        "requests": ['NPV']}


def get_bond_task(ccy, curve, asof, pmnt_date):
    """

    :str ccy: 'EUR'
    :dict curve : example -> please execute '??service' command for curve example
    :datetime asof: datetime(2018, 5, 12)
    :datetime pmnt_date: datetime(2019, 5, 12)
    """

    task = _get_task_form(ccy, curve, asof)
    task['fixedFlow'] = {"currency": ccy, "paymentDate": generate_date(pmnt_date)}
    return task


def get_libor_flow_task(ccy, dsc_curve, asof, tenor, fixingDate, paymentDate, libor_curve=None):
    """

    :str ccy: 'EUR'
    :curve dsc_curve: dict (check mkt data example)
    :datetime asof: datetime(2018, 5, 12)
    :int tenor: 6
    :datetime fixingDate: datetime(2018, 5, 12)
    :datetime paymentDate: datetime(2018, 5, 12)
    :None, float or dict libor_curve:
        if None: flat curve 0.01 * tenor
        if float: flat curve float_value
        if dict: user defined libor curve
    """

    task = _get_task_form(ccy, dsc_curve, asof)
    add_libor_curve(task, ccy, tenor, curve=libor_curve)
    task['liborFlow'] = {
              "liborIndex": f"{ccy}LIBOR{tenor}M",
              "fixingDate": generate_date(fixingDate),
              "paymentDate": generate_date(paymentDate)}
    return task


def get_swap_task(start, end, floatFreq, fixFreq, fixRate, spread, N, ccy, curve, asof):
    """
    :datetime start: start of swap
    :datetime end: end of swap
    :string floatFreq: float leg frequency ex. '6M'
    :string fixFreq: fix leg frequency ex. '1Y'
    :float fixRate: ex. 0.2
    :float spread: float margin, ex. 0.2
    :float N: ex. 10000.0
    :string ccy: ex. 'USD'
    :dict curve: please check service market data examples at the beginning
    :datetime asof: ex. datetime(2018, 3, 23)
    """
    swap_task = _get_task_form(ccy, curve, asof)
    swap_task['vanillaSwap'] = _get_swap(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N)
    return swap_task


def get_swo_task(start, end, floatFreq, fixFreq, fixRate, spread, N, ccy, curve, asof, expiry, smiles):
    """
    For previous parameters please refer to get_swap_task
    :string expiry: expity of swaption ex. '1Y'
    :dict smiles: volatiltiy curves, please check service market data examples at the beginning
    """
    swo_task = _get_task_form(ccy, curve, asof)
    swo_task['marketData'] = _get_swo_vol_mkt_data(ccy, curve, smiles)
    swo_task['diffusion'] = {"rates": {ccy: {"type": "BLACK_FORWARD"}}}
    swo_task['rateEuropeanSwaption'] = _get_swo(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N, expiry)
    return swo_task


def get_mco_task(start, end, floatFreq, fixFreq, fixRate, spread, N, ccy, curve, asof, expiry, smiles, correl, tenors, meanR, mco_proxy):
    """
    For previous parameters please refer to get_swo_task
    :float correl: correlation between two swap rates, ex. 0.5
    :list of string tenors: tenors of each swap rate. ex. ['1Y', '2Y']
    :float meanR: mean reversion parameter in Hull White model, ex. 0.1
    :int mco_proxy: type of mid curve approximation , ex. 1
    """
    mco_task = _get_task_form(ccy, curve, asof)
    mco_task['marketData'] = _get_swo_correl_mkt_data(ccy, curve, smiles, expiry, correl, tenors)
    mco_task['diffusion'] = {
                        "rates": {
                                ccy: {
                                    "type": "BLACK_TSR",
                                    "parameters": {
                                        "meanReversion": meanR,
                                        "midCurveApprox": mco_proxy
                                                  }
                                    }
                                }
                            }
    mco_task['settings'] = {
                    "pricingDate": asof,
                    "hardware": "CPU",
                    "numericalMethod": "COPULA",
                    "modelPrecision": "HUGE"
                }
    mco_task['midcurveSwaption'] = _get_swo(ccy, start, end, floatFreq, fixFreq, fixRate, spread, N, expiry)

    return mco_task


""" -----------------------------------------------TASK ADDERS-----------------------------------------------------"""

def add_swap_initial_stub(task, initial_stub_date, indexes=None):
    """

    :param initial_stub: datetime.datetime
    list indexes: ['EURLIBOR3M'] or ['EURLIBOR3M' , 'EURLIBOR6M']
    """

    assert 'vanillaSwap' in task, 'your task is not a swap task'

    start_date = task['vanillaSwap']['payerLeg']['schedule']['startDates'][0]

    task['vanillaSwap']['receiverLeg']['schedule']['fixingDates'] = [generate_date(initial_stub_date)] + task['vanillaSwap']['receiverLeg']['schedule']['fixingDates']
    for leg in ['payerLeg', 'receiverLeg']:
        task['vanillaSwap'][leg]['schedule']['startDates'] = [generate_date(initial_stub_date)] + task['vanillaSwap'][leg]['schedule']['startDates']

        for d in ['endDates', 'paymentDates']:
            task['vanillaSwap'][leg]['schedule'][d] = [start_date] + task['vanillaSwap'][leg]['schedule'][d]

    if indexes: task['vanillaSwap']['receiverLeg']['initialStub'] = indexes


def add_swap_final_stub(task, final_stub):
    NotImplemented('not implemented')



def add_swap_historical_fixing(task, ccy, tenor, date, value=None):
    if value is None: value = 0.1*tenor
    try:    # if any of fixings exists
        task['vanillaSwap']['historicalData']['rateFixings'][f'{ccy}LIBOR{tenor}M'] = {
                "dates": [generate_date(date)],
                "fixings": [value]
        }
    except KeyError:    # no fixings yet
        task['vanillaSwap']['historicalData'] = {
            'rateFixings': {f'{ccy}LIBOR{tenor}M':{
                "dates": [generate_date(date)],
                "fixings": [value]}}
        }


def add_libor_curve(task, ccy, tenor, curve=None):

    if curve is None or isinstance(curve, float):
        if curve is None:
            abs_value = 0.001 * tenor
        else:
            abs_value = curve
        curve = {
                            "dates": ["2018-01-01T00:00:00.000Z", "2023-01-01T00:00:00.000Z"],
                            "zeroRates": [abs_value] * 2}

    task['marketData']['rates'][f'{ccy}LIBOR{tenor}M'] = {'rateCurve': curve}


def change_fixRate(task, value):
    task['vanillaSwap']['payerLeg']['fixedRate'] = value


def change_rateIndex(task, ccy, tenor):
    task['vanillaSwap']['receiverLeg']['rateIndex'] = f'{ccy}LIBOR{tenor}M'


"""----------------------------------------UNDER DEV-------------------------------------------------------"""

#
# def get_cms_task(asof, ccy, curve, tenor, start, pmnt_date):
#     return {
#     'marketData': {
#         "rates": {
#             ccy: {'rateCurve': curve},
#             manual_fwd_label(ccy):  {'rateCurve': curve}
#
#         }
#     },
#
#     "cmsFlow": {
#       "cmsIndex": f'{ccy}CMS{tenor}' ,
#       "fixingDate":  start,
#       "paymentDate": pmnt_date,
#    },
#
#     "settings": {
#         "pricingDate": asof,
#         "hardware": "CPU",
#         "numericalMethod": "INTRINSIC"
#     },
#     "requests": ['NPV']
#     }
"""----------------------------------------DUMMIES-------------------------------------------------------"""

def xva_dummy_diffusion(asof):
    return {
        'marketData': {
            'rates': {
                'USD': {
                  'rateCurve': {
                    'dates': [ '0D', '1Y' ],
                    'zeroRates': [ 0.0, 0.0 ]
                   },
                   'capFloorVolatilities': { 
                     'liborTenors'    : ['3M'],
                     'optionExpiries' : ['2017-04-01T00:00:00.000Z', '2030-10-01T00:00:00.000Z'],
                     'strikes'        : [0.005, 0.01, 0.02],
                     'volatilities'   : [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                   }
                }
            }
        },
        'diffusion': {
            'rates': {
                'USD': {'type': 'HULL_AND_WHITE'}
            }
        },
        'requests': ['XVA_DIFFUSION'],
        'settings': {
            'pricingDate': asof,
            'hardware': 'CPU',
            'xva': {
                'currency': 'USD',
                'dates': ['0D', '1M', '2M']
            }
        }
    }