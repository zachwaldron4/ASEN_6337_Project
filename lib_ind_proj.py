import numpy as np
import pandas as pd

import os.path
from os import path
from scipy.io import loadmat  #allows us to read in .mat files

from datetime import datetime, timedelta



# data_path = 'D:/data/'
data_path = 'E:/data/'


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




def make_champ_timeseries(years, days):
    tleng = 0
    #     dens_sat_full = np.zeros(4*365*1900)
    time_full= []
    lon_full= []
    lat_full= []
    slt_full= []
    rho_full = []
    date = []

    time_last_index = []


    i = 0

    for iyear,year in enumerate(years):

        for iday,day in enumerate(days):
            champ, breakloop = get_CHAMP_data(year, day) 
            if breakloop == False:
                leng = np.size(champ,0)
        #         slt = np.zeros(leng)
                kp_indexed_sat = np.zeros(leng)

                # for isat, iheight in enumerate(champ['Height']):
                    # find the index of the model data that the satelite is closest to
                #     [time_index] = find_min_index(time_model, time_sat[isat])

                    # calculate the solar local time in the minimized coords

                #     kp_indexed_sat[isat] = data_model['Kp'][time_index][0]
                #             date.append( str(year)+'/'+ str(day) + '/' + str(time_sat[isat])[0])
                #             date.append(datetime(year, 1, 1) + timedelta(float(day) - 1) + timedelta( hours = time_sat[isat] )) 
                leng2 = leng+tleng
                if i == 0:
                     time_full[tleng:leng2] = champ['time_hours'][:leng]
                else:
                    time_full[tleng:leng2] = champ['time_hours'][:leng] # + time_last_index 

                slt_full[tleng:leng2]  = champ['LocTim'][:leng] 
                rho_full[tleng:leng2]  = champ['D400'][:leng] 
                lon_full[tleng:leng2]  = champ['Lon'][:leng] 
                lat_full[tleng:leng2]  = champ['Lat'][:leng] 

#                 time_last_index = time_full[-1]

    #                 time_last_index[iday, iyear] =  champ['time_hours'][-1]
                # Kp_model_full[:leng + tleng]   = kp_indexed_sat[:leng]      
                for it, itime in enumerate( time_full[tleng:leng2]):
#                     print(it,itime)
                    date.append(datetime(year, 1, 1) + timedelta(days = float(day), hours = itime) ) #+ timedelta( hours = time_sat[isat] )) 

            
                tleng = tleng + leng
#                 print(year,'/',day)
                
                i+=1
    #             print(np.shape(rho_full))
            elif breakloop == True:
                i+=1
                continue

            # END iday
        # END iyear
        
        df = pd.DataFrame(data ={'date':date  ,
                         'rho': rho_full,
                         'time_full':time_full,
                         'slt_full':slt_full, 
                         'lon_full':lon_full,
                         'lat_full':lat_full
                        } )

    return(df)




def orbit_average_density(file):
    champ = pd.read_pickle(file)

    lon = champ['lon_full'].values
    lat = champ['lat_full'].values
    rho = champ['rho'].values

    # Python:
    i = np.nonzero( lat[1:]*lat[0:-1]  <  np.logical_and(0 , lat[1:] > lat[0:-1] )  )
    i = i[0]

    D400_avg = np.zeros(np.size(i))

    # range(np.size(i)-1)
    for j in range(np.size(i)-1):
    #     print(j+1)
        D400_avg[j] = np.mean(  rho[   i[j] : i[j+1]-1  ]  );
    return(D400_avg)




