#Read_netcdf_variables

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os.path
from os import path
from scipy.io import loadmat  #allows us to read in .mat files
from scipy.stats import linregress 
# densityAvg_dailyCHAMP_day = np.zeros((np.size(days), np.size(years)))
# densityAvg_dailyCHAMP_night = np.zeros((np.size(days), np.size(years)))
from tabulate import tabulate


# data_path = 'D:/'
data_path = 'E:/data/'


# The below functin lets you save pandas DataFrame to a CSV
def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)
    
# pd.DataFrame.to_fwf = to_fwf(df_champ,'interp400_data/df_champ.txt')




def read_netcdf_variables( variables, year, day):
    
    path_model = 'E:/data/day_night_data/sres_mi_nonm/%d/' % year  
    filename = path_model+ 's_%03d.nc'% day  
    
    status = os.path.exists(filename)
    
    if status == True:
        data = {}
        for i, var_names in enumerate(variables):
            ncid =  Dataset(filename,"r+", format="NETCDF4")# filename must be a string
            varData = ncid.variables
            data[var_names] = np.array(varData[var_names])  
    elif status == False:
        print('No File:', day,'/', year,'N/A', filename )
        breakloop = True
        data = 0
        return( data , breakloop)
    breakloop = False
    return(data,breakloop )


# get_CHAMP_data checks if a file exists for that DOY
# and returns a Dataframe with all data
def get_CHAMP_data(year, day):
#     path_champ = 'D:/day_night_data/CHAMP_full_2002_2005/'
    path_champ = data_path + 'day_night_data/CHAMP_full_2002_2005/'
    filename = path_champ + '%d/Density_3deg_' % year + str(year)[-2:]  +'_%03d.mat' % day

    status = os.path.exists(filename)
    
    if status == True:
        data_champ = loadmat(filename)
    elif status == False:
        print('No File:', day,'/', year,'N/A', filename )
        breakloop = True
        df = 0
        return(df, breakloop)

    data_champ = loadmat(filename)

    time_sat = np.transpose(data_champ['Sec']['data'][0][0])[0]/3600 #in hours
    lon_sat = np.transpose(data_champ['Lon']['data'][0][0])[0]
    lat_sat = np.transpose(data_champ['Lat']['data'][0][0])[0]
    dens_sat =  np.transpose(data_champ['Density']['data'][0][0])[0]
    dens_sat_norm400 = np.transpose(data_champ['D400']['data'][0][0])[0]
    LocTim_sat = np.transpose(data_champ['LocTim']['data'][0][0])[0]
    height_sat =  np.transpose(data_champ['Height']['data'][0][0])[0]

    df = pd.DataFrame(data ={'LocTim':LocTim_sat  ,'D400': dens_sat_norm400, 'Density': dens_sat , 'time_hours': time_sat , 'Lon': lon_sat, 'Lat': lat_sat , 'Height': height_sat} )
#     print('Loaded day data loaded into pandas dataframe')
    breakloop = False
    return(df, breakloop)

def get_CHAMP_Mehta_data(year, day):
    path_champ = data_path + 'day_night_data/CHAMP_Mehta_2002_2005/'
    filename = path_champ + str(year) +'/CHAMP_Density_%02d_' %   int(str(year)[-2:])  +'%03d_v1.txt' % day

    status = os.path.exists(filename)
    
    if status == True:
        dataframe = pd.read_csv(filename, header = None, sep= '\s+', \
                   skiprows = 1,\
                   names = ['GPS Time (sec)','Geodetic Altitude (km)','Geodetic Latitude (deg)', 'Geodetic Longitude (deg)',\
                           'Local Solar Time (hours)',\
                            'Velocity Magnitude (m/s)',\
                            'Surface Temperature (K)',\
                            'Free Stream Temperature (K)',\
                            'Yaw (rad)',\
                            'Pitch (rad)',\
                            'Proj_Area_Eric (m^2)',\
                            'CD_Eric (~)',\
                            'Density_Eric (kg/m^3)',\
                            'Proj_Area_New (m^2)',\
                            'CD_New (~)',\
                            'Density_New (kg/m^3)',\
                           ] )    
        
        dataframe['time_hours'] = dataframe['GPS Time (sec)']/3600
    elif status == False:
        print('No File:', day,'/', year,'N/A', filename )
        breakloop = True
        df = 0
        return(df, breakloop)
    breakloop = False
    return(dataframe, breakloop)

def find_min_index(model , satellite):
    min_array = np.absolute(model - satellite)
    min_val = np.amin(np.absolute(model - satellite))
    index = np.where(min_array == min_val)
    return(index)


def logic_masks(leng, slt, lat_sat, kp_indexed_sat, dens_sat_norm400, iday, iyear ):
    
    mask_day = np.logical_and(slt[:leng] >= 10.5, slt[:leng] <= 16.5)
    mask_night = np.logical_or((slt[:leng] >= 22.5), (slt[:leng] <= 4.5))  
    mask_lats =  np.logical_and(lat_sat[:leng] <=30 , lat_sat[:leng] >= -30)
    mask_kp =   np.logical_not(kp_indexed_sat >= 3)   

    leng_logic_day = np.sum(mask_day & mask_lats & mask_kp)
    leng_logic_night = np.sum(mask_night & mask_lats & mask_kp)

    densityAvg_dailyCHAMP_day = np.mean(dens_sat_norm400[ mask_day & mask_lats & mask_kp]);
    densityAvg_dailyCHAMP_night = np.mean(dens_sat_norm400[ mask_night & mask_lats & mask_kp]);

    return(leng_logic_day, leng_logic_night, densityAvg_dailyCHAMP_day, densityAvg_dailyCHAMP_night) 






# def get_ap_values(filename):
# #     filename = 'E:/Day_Night_study/day_night_data/2002'

# #     print(ap['data'][1])
#     ap = pd.read_csv(filename,  names = ['data'] )
# #     year_ap = ap['data'][1][0:2]     # columns 1- 2     I2    YEAR
# #     print(year_ap)

# #     month_ap = ap['data'][1][2:4]   # columns 3- 4     I2    MONTH
# #     print(month_ap)

# #     day_ap = ap['data'][1][4:6]     # columns 5- 6     I2    DAY
# #     print(day_ap)


# #     #56-58     I3    Ap or PLANETARY EQUIVALENT DAILY AMPLITUDE--the arithmetic mean of the day's eight ap values.
# #     ap_mean = ap['data'][1][55:58]
# #     print(ap_mean)

#     ap_values = np.zeros((np.size(ap['data'])))

#     for i,values in enumerate(ap['data']):
#         ap_values[i] = ap['data'][i][55:58]

# #     print(ap_values.shape)
#     return(ap_values)

def get_ap_values(filename):
    ap = pd.read_csv(filename,  names = ['data'] )
    ap_values = np.zeros((np.size(ap['data'])))
   
    for i,values in enumerate(ap['data']):
        ap_values[i] = ap['data'][i][55:58]
    
    return(ap_values)


def get_f107_values(filename):

    f107_adjusted = pd.read_csv(filename, 
                         skiprows = 1 ,
                         names = ['time', 'f107'],
                         parse_dates = ['time']
                          )

    return(f107_adjusted)


def memory_concat(sltDay_long_matrix , sltNight_long_matrix):

    slt_day = np.concatenate(sltDay_long_matrix)
    slt_night = np.concatenate(sltNight_long_matrix)

    return(slt_day , slt_night)



#USE THIS CODE TO PICKLE THE DATA
#------------------------------------------------------------------------------
# noaa_2002_2010 = get_all_noaa()
# noaa_2002_2010.to_pickle(data_path + 'day_night_data/noaa_2002_2010_pickle')
#------------------------------------------------------------------------------


def get_NOAA_dataframe():
#     filename2001 = 'D:/day_night_data/2001'
#     filename2002 = 'D:/day_night_data/2002'
#     filename2003 = 'D:/day_night_data/2003'
#     filename2004 = 'D:/day_night_data/2004'
#     filename2005 = 'D:/day_night_data/2005'
#     filename2006 = 'D:/day_night_data/2006'
    filename2001 = data_path + 'day_night_data/2001'
    filename2002 = data_path + 'day_night_data/2002'
    filename2003 = data_path + 'day_night_data/2003'
    filename2004 = data_path + 'day_night_data/2004'
    filename2005 = data_path + 'day_night_data/2005'
    filename2006 = data_path + 'day_night_data/2006'
    filename2007 = data_path + 'day_night_data/2007'
    filename2008 = data_path + 'day_night_data/2008'
    filename2009 = data_path + 'day_night_data/2009'
    filename2010 = data_path + 'day_night_data/2010'
    filename2011 = data_path + 'day_night_data/2011'
    
    
    
    
    readfile2001 = pd.read_csv(filename2001 ,  names = ['data'] ) 
    df2001 = pd.DataFrame(data ={ 'data' : readfile2001['data'].values  })

    readfile2002 = pd.read_csv(filename2002 ,  names = ['data'] ) 
    df2002 = pd.DataFrame(data ={ 'data' : readfile2002['data'].values  }) 

    readfile2003 = pd.read_csv(filename2003 ,  names = ['data'] ) 
    df2003 = pd.DataFrame(data ={ 'data' : readfile2003['data'].values  }) 

    readfile2004 = pd.read_csv(filename2004 ,  names = ['data'] ) 
    df2004 = pd.DataFrame(data ={ 'data' : readfile2004['data'].values  }) 

    readfile2005 = pd.read_csv(filename2005 ,  names = ['data'] ) 
    df2005 = pd.DataFrame(data ={ 'data' : readfile2005['data'].values  }) 

    readfile2006 = pd.read_csv(filename2006 ,  names = ['data'] ) 
    df2006 = pd.DataFrame(data ={ 'data' : readfile2006['data'].values  }) 
    
    readfile2007 = pd.read_csv(filename2007 ,  names = ['data'] ) 
    df2007 = pd.DataFrame(data ={ 'data' : readfile2007['data'].values  }) 
    
    readfile2008 = pd.read_csv(filename2008 ,  names = ['data'] ) 
    df2008 = pd.DataFrame(data ={ 'data' : readfile2008['data'].values  }) 
    
    readfile2009 = pd.read_csv(filename2009 ,  names = ['data'] ) 
    df2009 = pd.DataFrame(data ={ 'data' : readfile2009['data'].values  }) 
    
    readfile2010 = pd.read_csv(filename2010 ,  names = ['data'] ) 
    df2010 = pd.DataFrame(data ={ 'data' : readfile2010['data'].values  }) 
    
    readfile2011 = pd.read_csv(filename2011 ,  names = ['data'] ) 
    df2011 = pd.DataFrame(data ={ 'data' : readfile2011['data'].values  }) 
    
    
    # print(df['data'][0][0][0:2])
    # filename2001= pd.DataFrame
    #     filename2002 = pd.read_csv('E:/Day_Night_study/day_night_data/2002',  names = ['data'] )
    #     filename2003 = pd.read_csv('E:/Day_Night_study/day_night_data/2003',  names = ['data'] )
    #     filename2004 = pd.read_csv('E:/Day_Night_study/day_night_data/2004',  names = ['data'] )
    #     filename2005 = pd.read_csv('E:/Day_Night_study/day_night_data/2005',  names = ['data'] )
    #     filename2006 = pd.read_csv('E:/Day_Night_study/day_night_data/2006',  names = ['data'] )
    # print(filename2001['data'])

    # years = [2001, 2002, 2003, 2004, 2005, 2006]
    # dfs = ['df2001["data"]' , 'df2002["data"]' , 'df2003["data"]', 'df2004["data"]', 'df2005["data"]' , 'df2006["data"]' ] 


    for i, val in enumerate(df2001['data']): 
        year = df2001['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2001['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2001['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2001.loc[i,['date'][0]] = pd.to_datetime(date_str)

    for i, val in enumerate(df2002['data']): 
        year = df2002['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2002['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2002['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2002.loc[i,['date'][0]] = pd.to_datetime(date_str)


    for i, val in enumerate(df2003['data']): 
        year = df2003['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2003['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2003['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2003.loc[i,['date'][0]] = pd.to_datetime(date_str)


    for i, val in enumerate(df2004['data']): 
        year = df2004['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2004['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2004['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2004.loc[i,['date'][0]] = pd.to_datetime(date_str)

    for i, val in enumerate(df2005['data']): 
        year = df2005['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2005['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2005['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2005.loc[i,['date'][0]] = pd.to_datetime(date_str)

    for i, val in enumerate(df2006['data']): 
        year = df2006['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2006['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2006['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2006.loc[i,['date'][0]] = pd.to_datetime(date_str)


    for i, val in enumerate(df2007['data']): 
        year = df2007['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2007['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2007['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2007.loc[i,['date'][0]] = pd.to_datetime(date_str) 

    for i, val in enumerate(df2008['data']): 
        year = df2008['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2008['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2008['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2008.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
    for i, val in enumerate(df2009['data']): 
        year = df2009['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2009['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2009['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2009.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
    for i, val in enumerate(df2010['data']): 
        year = df2010['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2010['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2010['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2010.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
    for i, val in enumerate(df2011['data']): 
        year = df2011['data'][i][0:2]     # columns 1- 2     I2    YEAR
        month = df2011['data'][i][2:4]    # columns 3- 4     I2    MONTH
        day = df2011['data'][i][4:6]      # columns 5- 6     I2    DAY
        date_str = month + '/' + day + '/' + year
        df2011.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        


    test_date =  np.concatenate([df2001['date'].values, df2002['date'].values , df2003['date'].values, df2004['date'].values, df2005['date'].values, df2006['date'].values, df2007['date'].values, df2008['date'].values, df2009['date'].values, df2010['date'].values, df2011['date'].values   ]) 
    test_data =  np.concatenate([df2001['data'].values, df2002['data'].values , df2003['data'].values, df2004['data'].values, df2005['data'].values, df2006['data'].values, df2007['data'].values, df2008['data'].values, df2009['data'].values, df2010['data'].values, df2011['data'].values]) 

    all_noaa = pd.DataFrame(data = {'Dates': test_date, 'values':test_data } )
#     print(all_noaa)
    return(all_noaa)



def isBlank (myString):
    return(not (myString and myString.strip()))

def isNotBlank (myString):
    return(bool(myString and myString.strip()))

def get_all_noaa():
    noaa = get_NOAA_dataframe()
    noaa = pd.DataFrame(data = {'Dates': noaa['Dates'], 'values':noaa['values'],'f107d':np.zeros(np.size(noaa['values'])), 'Ap':np.zeros(np.size(noaa['values'])), 'f107a':np.zeros(np.size(noaa['values'])),  'p107':np.zeros(np.size(noaa['values']))  } )

    for i,val in enumerate(noaa['Dates']):
        noaa.loc[i,['f107d']] = noaa['values'][i][65:70] # 66-70 
        noaa.loc[i,['Ap']] = noaa['values'][i][55:58]

        BlankTest = isNotBlank(noaa['f107d'][i])
#         print(i, val , noaa['f107d'][i])
        if BlankTest == True:
            pass
        elif BlankTest == False:
            noaa['f107d'][i] = np.nan

    l = np.size(noaa['f107d'])
    for ii , val in enumerate(noaa['Dates']):
        window = noaa['f107d'][ii-40:ii+41].values
#         print(ii, val ,'\n', noaa['f107d'][ii])

        add_vals = np.nansum(np.float64(window))
        size_vals = np.size(window)
        f107a = add_vals/size_vals
        noaa.loc[ii,['f107a']] = f107a

    for i,val in enumerate(noaa['Dates']):
        noaa.loc[i,['p107']] = (float(noaa['f107a'][i]) +  float(noaa['f107d'][i]) )/ 2


    noaa_2002_2010 = pd.DataFrame( data = {'Dates':noaa['Dates'][365:-365],  \
                                           'f107a' : noaa[365:-365]['f107a'],\
                                           'f107d' : noaa[365:-365]['f107d'],\
                                           'p107' : noaa[365:-365]['p107'],\
                                           'Ap' : noaa[365:-365]['Ap'] })
    noaa_2002_2010['datetime'] = pd.to_datetime(noaa_2002_2010['Dates'])
    noaa_2002_2010 = noaa_2002_2010.set_index('datetime')
    return(noaa_2002_2010)


# def get_NOAA_dataframe():
# #     filename2001 = 'D:/day_night_data/2001'
# #     filename2002 = 'D:/day_night_data/2002'
# #     filename2003 = 'D:/day_night_data/2003'
# #     filename2004 = 'D:/day_night_data/2004'
# #     filename2005 = 'D:/day_night_data/2005'
# #     filename2006 = 'D:/day_night_data/2006'
#     filename2001 = data_path + 'day_night_data/2001'
#     filename2002 = data_path + 'day_night_data/2002'
#     filename2003 = data_path + 'day_night_data/2003'
#     filename2004 = data_path + 'day_night_data/2004'
#     filename2005 = data_path + 'day_night_data/2005'
#     filename2006 = data_path + 'day_night_data/2006'
#     filename2007 = data_path + 'day_night_data/2007'
#     filename2008 = data_path + 'day_night_data/2008'
#     filename2009 = data_path + 'day_night_data/2009'
#     filename2010 = data_path + 'day_night_data/2010'
#     filename2011 = data_path + 'day_night_data/2011'
    
    
    
    
#     readfile2001 = pd.read_csv(filename2001 ,  names = ['data'] ) 
#     df2001 = pd.DataFrame(data ={ 'data' : readfile2001['data'].values  })

#     readfile2002 = pd.read_csv(filename2002 ,  names = ['data'] ) 
#     df2002 = pd.DataFrame(data ={ 'data' : readfile2002['data'].values  }) 

#     readfile2003 = pd.read_csv(filename2003 ,  names = ['data'] ) 
#     df2003 = pd.DataFrame(data ={ 'data' : readfile2003['data'].values  }) 

#     readfile2004 = pd.read_csv(filename2004 ,  names = ['data'] ) 
#     df2004 = pd.DataFrame(data ={ 'data' : readfile2004['data'].values  }) 

#     readfile2005 = pd.read_csv(filename2005 ,  names = ['data'] ) 
#     df2005 = pd.DataFrame(data ={ 'data' : readfile2005['data'].values  }) 

#     readfile2006 = pd.read_csv(filename2006 ,  names = ['data'] ) 
#     df2006 = pd.DataFrame(data ={ 'data' : readfile2006['data'].values  }) 
    
#     readfile2007 = pd.read_csv(filename2007 ,  names = ['data'] ) 
#     df2007 = pd.DataFrame(data ={ 'data' : readfile2007['data'].values  }) 
    
#     readfile2008 = pd.read_csv(filename2008 ,  names = ['data'] ) 
#     df2008 = pd.DataFrame(data ={ 'data' : readfile2008['data'].values  }) 
    
#     readfile2009 = pd.read_csv(filename2009 ,  names = ['data'] ) 
#     df2009 = pd.DataFrame(data ={ 'data' : readfile2009['data'].values  }) 
    
#     readfile2010 = pd.read_csv(filename2010 ,  names = ['data'] ) 
#     df2010 = pd.DataFrame(data ={ 'data' : readfile2010['data'].values  }) 
    
#     readfile2011 = pd.read_csv(filename2011 ,  names = ['data'] ) 
#     df2011 = pd.DataFrame(data ={ 'data' : readfile2011['data'].values  }) 
    
    
#     # print(df['data'][0][0][0:2])
#     # filename2001= pd.DataFrame
#     #     filename2002 = pd.read_csv('E:/Day_Night_study/day_night_data/2002',  names = ['data'] )
#     #     filename2003 = pd.read_csv('E:/Day_Night_study/day_night_data/2003',  names = ['data'] )
#     #     filename2004 = pd.read_csv('E:/Day_Night_study/day_night_data/2004',  names = ['data'] )
#     #     filename2005 = pd.read_csv('E:/Day_Night_study/day_night_data/2005',  names = ['data'] )
#     #     filename2006 = pd.read_csv('E:/Day_Night_study/day_night_data/2006',  names = ['data'] )
#     # print(filename2001['data'])

#     # years = [2001, 2002, 2003, 2004, 2005, 2006]
#     # dfs = ['df2001["data"]' , 'df2002["data"]' , 'df2003["data"]', 'df2004["data"]', 'df2005["data"]' , 'df2006["data"]' ] 


#     for i, val in enumerate(df2001['data']): 
#         year = df2001['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2001['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2001['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2001.loc[i,['date'][0]] = pd.to_datetime(date_str)

#     for i, val in enumerate(df2002['data']): 
#         year = df2002['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2002['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2002['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2002.loc[i,['date'][0]] = pd.to_datetime(date_str)


#     for i, val in enumerate(df2003['data']): 
#         year = df2003['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2003['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2003['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2003.loc[i,['date'][0]] = pd.to_datetime(date_str)


#     for i, val in enumerate(df2004['data']): 
#         year = df2004['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2004['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2004['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2004.loc[i,['date'][0]] = pd.to_datetime(date_str)

#     for i, val in enumerate(df2005['data']): 
#         year = df2005['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2005['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2005['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2005.loc[i,['date'][0]] = pd.to_datetime(date_str)

#     for i, val in enumerate(df2006['data']): 
#         year = df2006['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2006['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2006['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2006.loc[i,['date'][0]] = pd.to_datetime(date_str)


#     for i, val in enumerate(df2007['data']): 
#         year = df2007['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2007['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2007['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2007.loc[i,['date'][0]] = pd.to_datetime(date_str) 

#     for i, val in enumerate(df2008['data']): 
#         year = df2008['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2008['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2008['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2008.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
#     for i, val in enumerate(df2009['data']): 
#         year = df2009['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2009['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2009['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2009.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
#     for i, val in enumerate(df2010['data']): 
#         year = df2010['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2010['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2010['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2010.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        
#     for i, val in enumerate(df2011['data']): 
#         year = df2011['data'][i][0:2]     # columns 1- 2     I2    YEAR
#         month = df2011['data'][i][2:4]    # columns 3- 4     I2    MONTH
#         day = df2011['data'][i][4:6]      # columns 5- 6     I2    DAY
#         date_str = month + '/' + day + '/' + year
#         df2011.loc[i,['date'][0]] = pd.to_datetime(date_str) 
        


#     test_date =  np.concatenate([df2001['date'].values, df2002['date'].values , df2003['date'].values, df2004['date'].values, df2005['date'].values, df2006['date'].values, df2007['date'].values, df2008['date'].values, df2009['date'].values, df2010['date'].values, df2011['date'].values   ]) 
#     test_data =  np.concatenate([df2001['data'].values, df2002['data'].values , df2003['data'].values, df2004['data'].values, df2005['data'].values, df2006['data'].values, df2007['data'].values, df2008['data'].values, df2009['data'].values, df2010['data'].values, df2011['data'].values]) 

#     all_noaa = pd.DataFrame(data = {'Dates': test_date, 'values':test_data } )
# #     print(all_noaa)
#     return(all_noaa)


# def get_all_noaa():
#     noaa = get_NOAA_dataframe()
#     noaa = pd.DataFrame(data = {'Dates': noaa['Dates'], 'values':noaa['values'],'f107d':np.zeros(np.size(noaa['values'])), 'Ap':np.zeros(np.size(noaa['values'])), 'f107a':np.zeros(np.size(noaa['values'])),  'p107':np.zeros(np.size(noaa['values']))  } )

#     for i,val in enumerate(noaa['Dates']):
#         noaa.loc[i,['f107d']] = noaa['values'][i][65:70] # 66-70 
#         noaa.loc[i,['Ap']] = noaa['values'][i][55:58]
#     l = np.size(noaa['f107d'])
#     for i , val in enumerate(noaa['Dates']):
        
# #         print(i, val)
#         print(l, i, val)
#         if i > 40:
#             window_back = noaa['f107d'][i-40:i].values
#             window_forw = noaa['f107d'][i+1:i+41].values
#             add_vals = np.sum(np.float64(window_back)) + np.float64(noaa['f107d'][i]) + np.sum(np.float64(window_forw))
#             size_vals = np.size(window_forw) + np.size(window_back) + np.size(np.float64(noaa['f107d'][i]))
#             f107a = add_vals/size_vals
#             noaa.loc[i,['f107a']] = f107a
            
#         if i < (l - 40):
# #             window_forw = noaa['f107d'][i:i+40].values
# #             window_back = noaa['f107d'][i-40:i].values
#             window_forw = noaa['f107d'][i:i+40].values
#             add_vals = np.float64(noaa['f107d'][i]) + np.sum(np.float64(window_forw))
#             size_vals = np.size(window_forw)  + np.size(np.float64(noaa['f107d'][i]))
#             f107a = add_vals/size_vals
#             noaa.loc[i,['f107a']] = f107a
            
            
            
            
#     for i,val in enumerate(noaa['Dates']):
#         noaa.loc[i,['p107']] = (float(noaa['f107a'][i]) +  float(noaa['f107d'][i]) )/ 2
       
    
#     noaa_2002_2010 = pd.DataFrame( data = {'Dates':noaa['Dates'][365:-365],  \
#                                            'f107a' : noaa[365:-365]['f107a'],\
#                                            'f107d' : noaa[365:-365]['f107d'],\
#                                            'p107' : noaa[365:-365]['p107'],\
#                                            'Ap' : noaa[365:-365]['Ap'] })
#     noaa_2002_2010['datetime'] = pd.to_datetime(noaa_2002_2010['Dates'])
#     noaa_2002_2010 = noaa_2002_2010.set_index('datetime')
#     return(noaa_2002_2010)

def make_DENdf_and_avgdf(i, df_champ, df_tiegcm, date_index, noaa, averages_champ, averages_tiegcm,DEN_champ, DEN_tiegcm):
    
    
    DEN_champ.loc[i,['Days']] =  df_champ['DensityDays']
    DEN_champ.loc[i,['Nights']] =  df_champ['DensityNights']
    DEN_champ.loc[i,['Days_Masked']] =  df_champ['DensityDays_Masked']
    DEN_champ.loc[i,['Nights_Masked']] =  df_champ['DensityNights_Masked']

    DEN_tiegcm.loc[i,['Days']] =  df_tiegcm['DensityDays']
    DEN_tiegcm.loc[i,['Nights']] =  df_tiegcm['DensityNights']
    DEN_tiegcm.loc[i,['Days_Masked']] =  df_tiegcm['DensityDays_Masked']
    DEN_tiegcm.loc[i,['Nights_Masked']] =  df_tiegcm['DensityNights_Masked']


    averages_champ.loc[i,['Date']] = date_index
    averages_champ.loc[i,['p107']] =   float(noaa['p107'][date_index])

    averages_champ.loc[i,['DayAverages']] =  [np.mean(df_champ['DensityDays'])]
    averages_champ.loc[i,['NightAverages']] = [np.mean(df_champ['DensityNights'])] 
    averages_champ.loc[i,['ratio']] =   [np.mean(df_champ['DensityDays']) / np.mean(df_champ['DensityNights'])] 

    averages_champ.loc[i,['DayAverages_Masked']] =  [np.mean(df_champ['DensityDays_Masked'])]
    averages_champ.loc[i,['NightAverages_Masked']] = [np.mean(df_champ['DensityNights_Masked'])] 
    averages_champ.loc[i,['ratio_Masked']] =   [np.mean(df_champ['DensityDays_Masked']) / np.mean(df_champ['DensityNights_Masked'])]         

    averages_tiegcm.loc[i,['Date']] = date_index
    averages_tiegcm.loc[i,['p107']] =   float(noaa['p107'][date_index])

    averages_tiegcm.loc[i,['DayAverages']] =  [np.nanmean(df_tiegcm['DensityDays'])]
    averages_tiegcm.loc[i,['NightAverages']] = [np.nanmean(df_tiegcm['DensityNights'])] 
    averages_tiegcm.loc[i,['ratio']] =   [np.nanmean(df_tiegcm['DensityDays']) / np.nanmean(df_tiegcm['DensityNights'])] 

    averages_tiegcm.loc[i,['DayAverages_Masked']] =  [np.nanmean(df_tiegcm['DensityDays_Masked'])]
    averages_tiegcm.loc[i,['NightAverages_Masked']] = [np.nanmean(df_tiegcm['DensityNights_Masked'])] 
    averages_tiegcm.loc[i,['ratio_Masked']] =   [np.nanmean(df_tiegcm['DensityDays_Masked']) / np.nanmean(df_tiegcm['DensityNights_Masked'])]         
    return(averages_tiegcm, averages_champ)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2


def find_modelgrid_base_index(TIEGCM, lon_sat, lat_sat, time_sat):
    
#     """ This function locates the ."""

    lon_start = -180 # first longitude point of the model grid (deg)
    dlon = 5  # difference between adjacent longitude grid points (deg)
    index_lon0 = int(np.mod(np.floor( (lon_sat - lon_start )/dlon ), 72 ) ) 


    lat_start = -87.5 # first longitude point of the model grid (deg)
    dlat = 5 # difference between adjacent longitude grid points (deg)
    index_lat0 = int(np.mod(np.floor( (lat_sat - lat_start)/dlat ), 36 ) )

    time_start = 0 # first longitude point of the model grid (deg)
    dtime = 1 # difference between adjacent longitude grid points (deg)
    index_time = int((np.mod(np.floor( (time_sat - time_start)/dtime ), 24 ) ))

    if index_lon0 == 71:
        index_lon1 = 0
    else:
        index_lon1 = index_lon0 + 1

#     if index_lat0 == 35:
#         index_lat1 = 0
#     else:
#         index_lat1 = index_lat0 + 1
    if index_lat0 >= 35 :
        index_lat1 = 35
        NPole_Flag = True
        SPole_Flag = False 

    elif index_lat0 <= 0: 
        index_lat1 = 0
        SPole_Flag = True
        NPole_Flag = False
    else: 
        index_lat1 = index_lat0 + 1
        NPole_Flag = False 
        SPole_Flag = False 


    return(index_lon0, index_lat0, index_lon1, index_lat1, index_time,NPole_Flag, SPole_Flag )


#     plt.plot(lon_sat, lat_sat, 'o', color='red')
#     plt.plot(TIEGCM['lon'][index_lon0], TIEGCM['lat'][index_lat0], 'o', color='blue' )
#     plt.plot(TIEGCM['lon'][index_lon1], TIEGCM['lat'][index_lat1], 'o', color='green' )
#     plt.plot(TIEGCM['lon'][index_lon0], TIEGCM['lat'][index_lat1], 'o', color='yellow' )
#     plt.plot(TIEGCM['lon'][index_lon1], TIEGCM['lat'][index_lat0], 'o', color='pink' )

#     plt.xlabel('lon')
#     plt.ylabel('lat')


#     lon_start = -180 # first longitude point of the model grid (deg)
#     dlon = 5 # difference between adjacent longitude grid points (deg)
#     index_lon = int(np.transpose(np.mod(np.floor( (lon_sat - lon_start + dlon/2)/dlon ), 72 ) ))
# #     print(index_lon)


#     lat_start = -87.5 # first longitude point of the model grid (deg)
#     dlat = 5 # difference between adjacent longitude grid points (deg)
#     index_lat = int( np.transpose(np.mod(np.floor( (lat_sat - lat_start + dlat/2)/dlat ), 36 ) ))
# #     print(index_lat)

#     time_start = 0 # first longitude point of the model grid (deg)
#     dtime = 1 # difference between adjacent longitude grid points (deg)
#     index_time = int(np.transpose(np.mod(np.floor( (time_sat - time_start + dtime/2)/dtime ), 24 ) ))

  
#     if index_lon == 71:
#         index_lon = 0
#     else:
#         pass
#     if index_lat == 35:
#         index_lat = 0
#     else:
#         pass
    
    
#     lon_model_0 = TIEGCM['lon'][index_lon]
#     lat_model_0 = TIEGCM['lat'][index_lat]
#     ut = TIEGCM['ut'][index_time]

    
    
#     lon_model_p1 = TIEGCM['lon'][index_lon + 1]
#     lat_model_p1 = TIEGCM['lat'][index_lat + 1]
#     lon_model_m1 = TIEGCM['lon'][index_lon - 1]
#     lat_model_m1 = TIEGCM['lat'][index_lat - 1]

# #     print('satellite:',lon_sat,lat_sat)
# #     print('model:',lon_model_0 , lat_model_0)        

#     # print(np.abs(lon_sat - lon_model_0))
#     lon_iplus = (np.abs(lon_sat - lon_model_p1))
#     lon_iminus = (np.abs(lon_sat - lon_model_m1))
# #     print(lon_iplus ,'\n',lon_iminus )


#     # print('\n',np.abs(lat_sat - lat_model_0))
#     lat_iplus = (np.abs(lat_sat - lat_model_p1))
#     lat_iminus = (np.abs(lat_sat - lat_model_m1))
# #     print(lat_iplus ,'\n',lat_iminus )

#     lon0 = lon_model_0 
#     lat0 = lat_model_0 


#     if lon_iplus > lon_iminus:
#         lon1 = lon_model_m1
#     elif lon_iplus < lon_iminus:
#         lon1 = lon_model_p1

#     if lat_iplus > lat_iminus:
#         lat1 = lat_model_m1
#     elif lat_iplus < lat_iminus:
#         lat1 = lat_model_p1
#     return(lon0, lat0, lon1, lat1, ut)




# def find_modelgrid_base_index(TIEGCM, lon_sat, lat_sat, time_sat):
#     lon_start = -180 # first longitude point of the model grid (deg)
#     dlon = 5 # difference between adjacent longitude grid points (deg)
#     index_lon = int(np.transpose(np.mod(np.floor( (lon_sat - lon_start + dlon/2)/dlon ), 72 ) ))
# #     print(index_lon)


#     lat_start = -87.5 # first longitude point of the model grid (deg)
#     dlat = 5 # difference between adjacent longitude grid points (deg)
#     index_lat = int( np.transpose(np.mod(np.floor( (lat_sat - lat_start + dlat/2)/dlat ), 36 ) ))
# #     print(index_lat)

#     time_start = 0 # first longitude point of the model grid (deg)
#     dtime = 1 # difference between adjacent longitude grid points (deg)
#     index_time = int(np.transpose(np.mod(np.floor( (time_sat - time_start + dtime/2)/dtime ), 24 ) ))

  
#     if index_lon == 71:
#         index_lon = 0
#     else:
#         pass
#     if index_lat == 35:
#         index_lat = 0
#     else:
#         pass
    
    
#     lon_model_0 = TIEGCM['lon'][index_lon]
#     lat_model_0 = TIEGCM['lat'][index_lat]
#     ut = TIEGCM['ut'][index_time]
    
#     lon_model_p1 = TIEGCM['lon'][index_lon + 1]
#     lat_model_p1 = TIEGCM['lat'][index_lat + 1]
#     lon_model_m1 = TIEGCM['lon'][index_lon - 1]
#     lat_model_m1 = TIEGCM['lat'][index_lat - 1]

# #     print('satellite:',lon_sat,lat_sat)
# #     print('model:',lon_model_0 , lat_model_0)        

#     # print(np.abs(lon_sat - lon_model_0))
#     lon_iplus = (np.abs(lon_sat - lon_model_p1))
#     lon_iminus = (np.abs(lon_sat - lon_model_m1))
# #     print(lon_iplus ,'\n',lon_iminus )


#     # print('\n',np.abs(lat_sat - lat_model_0))
#     lat_iplus = (np.abs(lat_sat - lat_model_p1))
#     lat_iminus = (np.abs(lat_sat - lat_model_m1))
# #     print(lat_iplus ,'\n',lat_iminus )

#     lon0 = lon_model_0 
#     lat0 = lat_model_0 


#     if lon_iplus > lon_iminus:
#         lon1 = lon_model_m1
#         lonindex_1 = index_lon - 1 
#     elif lon_iplus < lon_iminus:
#         lon1 = lon_model_p1
#         lonindex_1 = index_lon + 1 

#     if lat_iplus > lat_iminus:
#         lat1 = lat_model_m1
#         latindex_1 = index_lat - 1 
#     elif lat_iplus < lat_iminus:
#         lat1 = lat_model_p1
#         latindex_1 = index_lat + 1 
        
#     lonindex_0 = index_lon
#     latindex_0 = index_lat
#     utindex = index_time
    
#     return(lonindex_0, latindex_0, lonindex_1, latindex_1, utindex)


def DEN_and_ZG_vertprofs(TIEGCM, lon_sat, lat_sat, time_sat):

#     lon0, lat0, lon1, lat1, ut = find_modelgrid_base(TIEGCM, lon_sat, lat_sat, time_sat)
    indexlon0, indexlat0, indexlon1, indexlat1, indexut, NPole_Flag, SPole_Flag = find_modelgrid_base_index(TIEGCM, lon_sat, lat_sat, time_sat)
#     print(indexlon0, indexlon1, indexlat0, indexlat1)
#     print(TIEGCM['lon'][indexlon0], TIEGCM['lon'][indexlon1] , TIEGCM['lat'][indexlat0], TIEGCM['lat'][indexlat1])

#     print('NorthPole:', NPole_Flag ,'\nSouthPole:', SPole_Flag) 

#     if NPole_Flag == True: 
#         DEN_prof_ilon = np.zeros((np.size(TIEGCM['ilev'])  , (np.size(TIEGCM['lon'])))) 

#         for ilon, val in enumerate(TIEGCM['lon']):
# #             print(ilon)
# #             print(TIEGCM['DEN'][indexut, :, indexlat0, ilon])
#             DEN_prof_ilon[:,ilon] = TIEGCM['DEN'][indexut, :, indexlat0, ilon]
#              for i, val in enumerate(DEN_prof_ilon[ilon]):
#         print(i, val)

#         print(DEN_prof_ilon)
        
        
        
    grid_vals = pd.DataFrame(data = {'lon0' : TIEGCM['lon'][indexlon0] ,\
                                 'lonindex0': indexlon0 ,\
                                 'lon1'     :TIEGCM['lon'][indexlon1] ,\
                                 'lonindex1':indexlon1 ,\
                                 'lat0'     :TIEGCM['lat'][indexlat0] ,\
                                 'latindex0':indexlat0 ,\
                                 'lat1'     :TIEGCM['lat'][indexlat1] ,\
                                 'latindex1':indexlat1  }, index=[0])
    

    DEN_prof00 = TIEGCM['DEN'][indexut, :, indexlat0, indexlon0]
    DEN_prof10 = TIEGCM['DEN'][indexut, :, indexlat1, indexlon0]
    DEN_prof01 = TIEGCM['DEN'][indexut, :, indexlat0, indexlon1]
    DEN_prof11 = TIEGCM['DEN'][indexut, :, indexlat1, indexlon1]
    

    ZG_prof00 = TIEGCM['ZG'][indexut, :, indexlat0, indexlon0]
    ZG_prof10 = TIEGCM['ZG'][indexut, :, indexlat1, indexlon0]
    ZG_prof01 = TIEGCM['ZG'][indexut, :, indexlat0, indexlon1]
    ZG_prof11 = TIEGCM['ZG'][indexut, :, indexlat1, indexlon1]

    DEN_df = pd.DataFrame(data = {'00' :DEN_prof00 ,\
                                  '10' :DEN_prof10 ,\
                                  '01' :DEN_prof01 ,\
                                  '11' :DEN_prof11 ,\
                                })
    ZG_df = pd.DataFrame(data = {'00' :ZG_prof00 ,\
                                  '10' :ZG_prof10 ,\
                                  '01' :ZG_prof01 ,\
                                  '11' :ZG_prof11 ,\
                                })
    return(DEN_df, ZG_df, grid_vals)



def interp_to_altitude(DEN_df, ZG_df, height_sat):
    density_surf_at_height_sat = np.zeros(np.size(ZG_df.columns))
    i = 0
    for col in ZG_df.columns:
        xp = ZG_df[col][:].values*1e-5
        fp = DEN_df[col][:].values*1e3
        xval = height_sat
        density_surf_at_height_sat[i] =  np.exp( np.interp(xval, xp, np.log(fp))  )
        i += 1
    return(density_surf_at_height_sat)




def interp_lon_lat(TIEGCM, lon_sat, lat_sat, time_sat, height_sat):
    DEN_df, ZG_df, grid_vals = DEN_and_ZG_vertprofs(TIEGCM, lon_sat, lat_sat, time_sat)
    density_surf_at_height_sat = interp_to_altitude(DEN_df, ZG_df, height_sat)


    # print(grid_vals['lon0'].values[0])
    xval = lon_sat

    x_lon = [grid_vals['lon0'][0], grid_vals['lon1'][0]]
    f_den0 = [density_surf_at_height_sat[0], density_surf_at_height_sat[1]]
    interp_lon0 =   np.interp(xval, x_lon, f_den0  )
    # print(interp_lon0)

    f_den1 = [density_surf_at_height_sat[2], density_surf_at_height_sat[3]]
    interp_lon1 =   np.interp(xval, x_lon, f_den1  )
    # print(interp_lon1)

    density = interp_lon0 + ((lat_sat - grid_vals['lat0'][0]) / (grid_vals['lat1'][0] - grid_vals['lat0'][0]))*(interp_lon1 - interp_lon0)
    # print(density)
    return(density)




