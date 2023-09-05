
# timestamp manipulation 
from dateutil.relativedelta import relativedelta

def clean_temp(df,x,reference_years):
    """
    returns a temperaure column in celcius with missing values imputed;
    imputation is done with the average of the temperautes on the same
    day over all the reference years; division by 10 for celcius value
    """
    # if missing value occurs
    if x['Q_TG']==9:
        
        # list reference dates
        reference_dates = [x['DATE']+relativedelta(years=y) for y in reference_years]
        
        # mean temperatue over the references dates
        temp_value = df[df['DATE'].isin(reference_dates)]['TG'].mean()
        
        # division by 10 to convert to celcius value
        return int(temp_value)/10
    
    # else just division by 10 to convert to celcius value
    return x['TG']/10