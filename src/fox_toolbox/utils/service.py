from macs_service.date_util import generate_date, generate_dates


"""-----------------------------------------GET MARKET DATA-------------------------------------------------"""



def get_curve_mkt_data(ccy, curve):
    return {'rates': {
                        ccy: {'rateCurve': curve}
                    }
            }

def get_swo_vol_mkt_data(ccy, curve, smiles):
    swo_mkt = get_curve_mkt_data(ccy, curve)
    swo_mkt['rates'][ccy]['horizontalSwaptionVolatilities'] = smiles
    return swo_mkt

def get_swo_correl_mkt_data(ccy, curve, smiles, expiry, correl, tenors):
    swo_mkt = get_swo_vol_mkt_data(ccy, curve, smiles)
    swo_mkt['rates'][ccy]['swapTerminalCorrelations'] = {
                                        'correlations': [correl],
                                        'dates': [expiry],
                                        'swapTenors': list(tenors)
                                        }
    return swo_mkt


"""----------------------------------------GET PAYOFF----------------------------------------------------"""


def get_swap_simple(self):
    return  {
                 "nominal": self.nominal,
                 "startDate": self.start,
                 "fixedRate": self.strike,
                 "tenor": self.tenor,
                 "currency": self.ccy,
                "fixedRateDirection": "PAYER" if self.is_payer else "RECEIVER"
              }



def get_swap(start, end, floatFreq, fixFreq, fixRate, spread, N):
    assert start <= end, 'start > end'

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
            "paymentCurrency": "USD"
        },
        "receiverLeg": {
            "schedule": {
                "fixingDates": startDatesRec,
                "startDates": startDatesRec,
                "endDates": endDatesRec,
                "paymentDates": endDatesRec,
                "dayCountConvention": "ACT/365"
            },
            "rateIndex": "USDLIBOR" + floatFreq,
            "rateSpread": spread,
            "paymentCurrency": "USD"
        },
    }


def get_swo(start, end, floatFreq, fixFreq, fixRate, spread, N, expiry):
    return {
                "swap": get_swap(start, end, floatFreq, fixFreq, fixRate, spread, N),
                "exerciseDate": expiry
                }


"""-----------------------------------------------GET TASK--------------------------------------------------"""

def get_task(ccy, curve, asof):
    return {
        "marketData": {"rates": {ccy: {'rateCurve': curve, }}},
        "settings": {"pricingDate": generate_date(asof)},
        "requests": ['NPV']}


def get_bond_task(ccy, curve, asof, pmnt_date):
    task = get_task(ccy, curve, asof)
    task['fixedFlow'] = {"currency": ccy, "paymentDate": generate_date(pmnt_date)}
    return task


def get_libor_flow_task(ccy, dsc_curve, asof, tenor, fixingDate, paymentDate):
    "Libor curve is added with dummy values"

    task = get_task( ccy, dsc_curve, asof)
    add_libor_curve(task, ccy, tenor)
    task['liborFlow'] = {
              "liborIndex": f"USDLIBOR{tenor}M",
              "fixingDate": generate_date(fixingDate),
              "paymentDate": generate_date(paymentDate)}
    return task


def get_cms_task(self, pmnt_date):
    return {
    'marketData': {
        "rates": {
            self.ccy: {'rateCurve': self.curve},
            manual_fwd_label(self.ccy):  {'rateCurve': self.curve}

        }
    },

    "cmsFlow": {
      "cmsIndex": f'{self.ccy}CMS{self.tenor}' ,
      "fixingDate":  self.start,
      "paymentDate": pmnt_date,
   },

    "settings": {
        "pricingDate": self.asof,
        "hardware": "CPU",
        "numericalMethod": "INTRINSIC"
    },
    "requests": ['NPV']
    }


def get_swap_task(start, end, floatFreq, fixFreq, fixRate, spread, N, ccy, curve, asof):
    return {
                'marketData': get_curve_mkt_data(ccy, curve),
                "settings": {
                    "pricingDate": generate_date(asof),
                    "hardware": "CPU",
                    "numericalMethod": "INTRINSIC",
                    "modelPrecision": "SECOND_ORDER"
                },
                "requests": ['NPV'],
                "vanillaSwap": get_swap(start, end, floatFreq, fixFreq, fixRate, spread, N)
            }


def get_swo_task(self):
    task = {
        'marketData': get_swo_vol_mkt_data(self),
        "diffusion": {"rates": {self.ccy: {"type": "BLACK_FORWARD"}}},
        "settings": {
            "pricingDate": self.asof,
            "hardware": "CPU",
            "numericalMethod": "ANALYTICAL"
        },
        "requests": ['NPV'],
        "rateEuropeanSwaption": get_swo(self)
    }
    return task


def get_mco_task(self):
    return {
                'marketData': get_swo_correl_mkt_data(self),
                "diffusion": {
                        "rates": {
                                self.ccy: {
                                    "type": "BLACK_TSR",
                                    "parameters": {
                                        "meanReversion" : self.meanR,
                                        "midCurveApprox": self.mco_proxy
                                                  }
                                    }
                                }
                            },
                "settings": {
                    "pricingDate": self.asof,
                    "hardware": "CPU",
                    "numericalMethod": "COPULA",
                    "modelPrecision": "HUGE"
                },
                "requests": ['NPV'],
                "midcurveSwaption": get_swo(self)
            }


""" -----------------------------------------------ADDERS-----------------------------------------------------"""

def add_swap_initial_stub(task, initial_stub_date, indexes=None):
    """

    :param initial_stub: datetime.datetime
    """
    start_date = task['vanillaSwap']['payerLeg']['schedule']['startDates'][0]

    task['vanillaSwap']['receiverLeg']['schedule']['fixingDates'] = [generate_date(initial_stub_date)] + task['vanillaSwap']['receiverLeg']['schedule']['fixingDates']
    for leg in ['payerLeg', 'receiverLeg']:
        task['vanillaSwap'][leg]['schedule']['startDates'] = [generate_date(initial_stub_date)] + task['vanillaSwap'][leg]['schedule']['startDates']

        for d in ['endDates', 'paymentDates']:
            task['vanillaSwap'][leg]['schedule'][d] = [start_date] + task['vanillaSwap'][leg]['schedule'][d]

    if indexes: task['vanillaSwap']['receiverLeg']['initialStub'] = indexes


def add_swap_final_stub(task, final_stub):
    NotImplemented



def add_swap_historical_fixing(task, ccy, tenor, date, value=None):
    if value is None: value = 0.1*tenor
    try:    # if any of fixings exists
        task['vanillaSwap']['historicalData']['rateFixings'][f'{ccy}LIBOR{tenor}M'] = {
                "dates": [generate_date(date)],
                "fixings": [value]
        }
    except KeyError:
        task['vanillaSwap']['historicalData'] = {
            'rateFixings': {f'{ccy}LIBOR{tenor}M':{
                "dates": [generate_date(date)],
                "fixings": [value]}}
        }



def add_libor_curve(task, ccy, tenor, abs_value=None):
    """Add flat Libor curve value = 0.001 * tenor. Curve defined on 0D -> 5Y"""
    if abs_value is None: abs_value = 0.001*tenor
    task['marketData']['rates'][f'{ccy}LIBOR{tenor}M'] = {
        "rateSpreads": {
            "label": f"Rate curve: USD LIBOR {tenor}M",
            "dates": ["0D", "5Y"],
            "zeroRates": [abs_value - 0.01]*2}}

def change_fixRate(task, value=0):
    task['vanillaSwap']['payerLeg']['fixedRate'] = value

def change_rateIndex(task, ccy, tenor):
    task['vanillaSwap']['receiverLeg']['rateIndex'] = f'{ccy}LIBOR{tenor}M'


