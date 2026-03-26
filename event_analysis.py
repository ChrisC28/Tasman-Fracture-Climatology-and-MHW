import numpy as np
import scipy.signal as scipy_signal
from matplotlib import pyplot as plt
import pandas
import xarray

def phase_composite(da_to_composite,magnitude_phase_space,angle_phase_space,phase_partitions,phase_time,
                                    sensor_dims=['lat', 'lon'],sample_dim='time',magnitude_threshold=1):
    
    name = da_to_composite.name

    phase_composite = xarray.DataArray(np.zeros([phase_partitions.size-1,da_to_composite[sensor_dims[0]].size,da_to_composite[sensor_dims[1]].size]),
                                       coords=[('phase_angle',phase_partitions[0:-1]),
                                               (sensor_dims[0],da_to_composite[sensor_dims[0]].values), 
                                               (sensor_dims[1],da_to_composite[sensor_dims[1]].values)])
    
    phase_composite = phase_composite.to_dataset(name='phase_composite')
    phase_composite['N_samples'] =  xarray.DataArray(np.zeros([phase_partitions.size-1]),
                                                     coords=[('phase_angle',phase_partitions[0:-1])])
    
    sampling_dim = da_to_composite[sample_dim].values

    for i_phase in range(0,phase_partitions.size-1):
        idx_in_phase = np.nonzero(np.logical_and(magnitude_phase_space>magnitude_threshold,
                                             np.logical_and(angle_phase_space>phase_partitions[i_phase],angle_phase_space<phase_partitions[i_phase+1])) )
    
        binned_time_steps = phase_time[idx_in_phase]
        
        idx_time_steps_in_range = np.nonzero( np.logical_and(binned_time_steps>=sampling_dim.min(),binned_time_steps<=sampling_dim.max()) )[0]        
        binned_time_steps       = binned_time_steps[idx_time_steps_in_range]
        
        phase_composite['phase_composite'][i_phase,:,:] =  da_to_composite.sel({sample_dim:binned_time_steps},method='nearest').mean(dim=sample_dim)
        phase_composite['N_samples'][i_phase]   =  binned_time_steps.size
        
     
    if name != None:
        
        phase_composite = phase_composite.rename({'phase_composite':'phase_composite__'+name})
        phase_composite = phase_composite.rename({'N_samples':'N_samples__'+name})


    return phase_composite


def phase_space_event_indentification(x,y,time, initiation_threshold,phase_turning_angle_threshold,clockwise=True):
    
    magnitude_phase_space   = np.sqrt(x*x + y*y)
    angle_phase_space = np.rad2deg(np.arctan2(y,x))
    
    
    peaks, properties = scipy_signal.find_peaks(magnitude_phase_space, prominence=0.5,height=initiation_threshold)
    phase_valleys, properties = scipy_signal.find_peaks(-np.rad2deg(angle_phase_space), prominence=50)
    
    
    diff_from_initiation_threshold = magnitude_phase_space-initiation_threshold
    first_pass_initiation_indicies = np.nonzero(np.diff(np.sign(diff_from_initiation_threshold))==2)[0]
    first_pass_termination_indicies = np.nonzero(np.diff(np.sign(diff_from_initiation_threshold))==-2)[0]
    
    
    
    matching_initiation_indicies   = []
    matching_termination_indicies  = []

    for i_initiation_index in range(0,len(first_pass_initiation_indicies)):
        diff_termination_initiation = first_pass_termination_indicies-first_pass_initiation_indicies[i_initiation_index]
        if np.any(diff_termination_initiation>0):
        
            plus_array = [elem for elem in diff_termination_initiation if elem > 0]
            matching_index = np.nonzero(diff_termination_initiation==np.min(plus_array))[0][0]
            
            min_elem = first_pass_termination_indicies[matching_index]

            matching_termination_indicies.append(min_elem)
            matching_initiation_indicies.append(first_pass_initiation_indicies[i_initiation_index])
        
    

    
    event = {}
    event['initiation_index']  = []
    event['initiation_time']   = []
    event['termination_index'] = []
    event['termination_time']  = []
    event['peak_time']         = []
    event['peak_index']        = []
    
    event['event_duration']    = []

    
    plot_counter = 0
    for i_event in range(0,len(matching_initiation_indicies)):
        #print('Event:', i_event)
        
        if matching_termination_indicies[i_event]-matching_initiation_indicies[i_event]<3:
            continue 
            
        phase_angle_series = angle_phase_space[matching_initiation_indicies[i_event]:matching_termination_indicies[i_event]]
    
        diff_phase_angle_series = np.diff(phase_angle_series)
    
        idx_wrap = np.nonzero(np.abs(diff_phase_angle_series)>200)[0]
        
        diff_phase_angle_series[idx_wrap] = 0 
        
        new_phase_angle_series  = np.cumsum(np.concatenate([phase_angle_series[0:1],diff_phase_angle_series])) 
        
        
        
        linear_fit_coefficients = np.polyfit(np.arange(0,len(new_phase_angle_series),1), new_phase_angle_series, 1)
        linear_fit_to_phase     = linear_fit_coefficients[1] + linear_fit_coefficients[0]*np.arange(0,len(new_phase_angle_series),1)
        
        phase_slope      = linear_fit_coefficients[0]

        phase_difference = linear_fit_to_phase[-1]-linear_fit_to_phase[0]
        
        
        if  clockwise:
            phase_slope = -1*phase_slope
        
            
            
        if (phase_slope> 0)  and  np.abs(phase_difference)>phase_turning_angle_threshold:
            #if plot_counter<20:
            ##    plt.figure(plot_counter)
            #    plt.plot(magnitude_phase_space[matching_initiation_indicies[i_event]:matching_termination_indicies[i_event]+5])
            #    plot_counter = plot_counter+1
            #    plt.figure(plot_counter)
            #    plt.plot(new_phase_angle_series)
            #    plt.plot(linear_fit_to_phase,'r')
            #    plot_counter = plot_counter+1
            #if plot_counter>20:
            #    dsadda
            
            event['initiation_index'].append(matching_initiation_indicies[i_event])
            event['initiation_time'].append(time[matching_initiation_indicies[i_event]])
            
            event['termination_index'].append(matching_termination_indicies[i_event])
            event['termination_time'].append(time[matching_termination_indicies[i_event]])
            
            
            event['peak_index'].append( np.argmax(magnitude_phase_space[ matching_initiation_indicies[i_event]:matching_termination_indicies[i_event] ]) + 
                                        matching_initiation_indicies[i_event] )
            event['peak_time'].append( event['peak_index'][-1])
            
            event['event_duration'].append(( time[matching_termination_indicies[i_event]] - 
                                              time[matching_initiation_indicies[i_event]] )/np.timedelta64(1, 'D'))

            
    event['n_events'] = len(event['event_duration'])
    return event


def shifted_event_composite_average(event_dictionary,da_to_composite,magnitude_phase_space,angle_phase_space,n_shifts,shift_increments,
                                    sensor_dims=['lat', 'lon'],sample_dim='time',target_phase=0):

    event_duration = np.asarray(event_dictionary['event_duration'])

    event_initiation_times  = np.asarray(event_dictionary['initiation_time'])
    event_termination_times = np.asarray(event_dictionary['termination_time'])
    event_initiation_idx    = np.asarray(event_dictionary['initiation_index'])
    event_termination_idx   = np.asarray(event_dictionary['termination_index'])
    
    
    n_events = event_initiation_times.size
    n_before_after = 5

    time_steps_for_composite = []
    
    name = da_to_composite.name
    n_events_in_composite = 0
    for i_event in range(0,n_events):
        
        magnitude_for_event = magnitude_phase_space[event_initiation_idx[i_event]-n_before_after:event_termination_idx[i_event]+n_before_after]
        
        phase_for_event = angle_phase_space[event_initiation_idx[i_event]-n_before_after:event_termination_idx[i_event]+n_before_after]
        time_for_event  = da_to_composite[sample_dim][event_initiation_idx[i_event]-n_before_after:event_termination_idx[i_event]+n_before_after].values
        
        
        target_phase_crossing = np.diff(np.sign(phase_for_event-target_phase))
        
        if np.any(target_phase_crossing>=1):
            idx_target_phase_crossing = np.nonzero(target_phase_crossing>=1)[0]
        
            
            phase_difference_at_crossing = np.diff(phase_for_event)[idx_target_phase_crossing]
            
            wrap_around_indicies = np.nonzero(np.abs(phase_difference_at_crossing)>180)[0]
            if len(wrap_around_indicies) != 0:
                idx_target_phase_crossing  = np.delete(idx_target_phase_crossing,wrap_around_indicies)
        
            if len(idx_target_phase_crossing) !=0:  
                time_steps_for_composite.append(time_for_event[idx_target_phase_crossing[-1]])
                
                n_events_in_composite =n_events_in_composite+1
    time_steps_for_composite = np.asarray(time_steps_for_composite)
    
    
    time_shifts = np.linspace(-n_shifts*shift_increments,n_shifts*shift_increments,2*n_shifts+1)
    event_shifted_composite = xarray.DataArray(np.zeros([2*n_shifts+1,da_to_composite[sensor_dims[0]].size,da_to_composite[sensor_dims[1]].size]),
                                              coords=[('time_shift',time_shifts),
                                                      (sensor_dims[0],da_to_composite[sensor_dims[0]].values), 
                                                      (sensor_dims[1],da_to_composite[sensor_dims[1]].values)])
    
    start_date = da_to_composite[sample_dim][0].values
    end_date   = da_to_composite[sample_dim][-1].values
    for i_shift in range(0,2*n_shifts+1):
        
        shifted_times = np.asarray(time_steps_for_composite) + np.timedelta64(int(time_shifts[i_shift]),'D')
        shifted_times = shifted_times[shifted_times>start_date]
        shifted_times = shifted_times[shifted_times<end_date]
        
        event_shifted_composite[i_shift,:,:] = da_to_composite.sel({sample_dim:shifted_times},method='nearest').mean(dim=sample_dim)

    event_shifted_composite = event_shifted_composite.rename('shifted_composite') if name is None else event_shifted_composite.rename('shifted_composite__'+name)
    
    
    event_shifted_composite = event_shifted_composite.to_dataset()
    event_shifted_composite['n_events_in_composite'] = n_events_in_composite
    return event_shifted_composite

def shifted_initiation_composite_average(event_dictionary,da_to_composite,angle_phase_space,n_shifts,shift_increments,
                                         phase_partitions,sensor_dims=['lat', 'lon'],sample_dim='time'):
    
    time_shifts = np.linspace(-n_shifts*shift_increments,n_shifts*shift_increments,2*n_shifts+1)

    event_initiation_times  = np.asarray(event_dictionary['initiation_time'])
    event_initiation_idx    = np.asarray(event_dictionary['initiation_index'])

    phase_for_initiation_time = angle_phase_space[event_initiation_idx]
    
    start_date = da_to_composite[sample_dim][0].values
    end_date   = da_to_composite[sample_dim][-1].values
    
    initiation_event_shifted_composite = xarray.DataArray(np.zeros([len(phase_partitions)-1,2*n_shifts+1,da_to_composite[sensor_dims[0]].size,da_to_composite[sensor_dims[1]].size]),
                                         coords=[( 'phase_partitions',0.5*(phase_partitions[1::]+phase_partitions[0:-1]) ),
                                                 ( 'time_shift',time_shifts ),
                                                 ( sensor_dims[0],da_to_composite[sensor_dims[0]].values ), 
                                                 ( sensor_dims[1],da_to_composite[sensor_dims[1]].values )])
 
    
    for i_phase in range(0,phase_partitions.size-1):
        idx_in_phase = np.nonzero(np.logical_and(phase_for_initiation_time.values>phase_partitions[i_phase],
                                                 phase_for_initiation_time.values<phase_partitions[i_phase+1]))[0]
        
        time_for_initiation_in_phase = event_initiation_times[idx_in_phase]
        for i_shift in range(0,2*n_shifts+1):
        
            shifted_times = np.asarray(time_for_initiation_in_phase) + np.timedelta64(int(time_shifts[i_shift]),'D')

            shifted_times = shifted_times[shifted_times>start_date]
            shifted_times = shifted_times[shifted_times<end_date]
            
            initiation_event_shifted_composite[i_phase,i_shift,:,:] = da_to_composite.sel({sample_dim:shifted_times}).mean(dim=sample_dim)

        
    return initiation_event_shifted_composite



def random_sampling_for_quantiles(N_samples,da,sensor_dims=['lat', 'lon'],sample_dim='time',N_monte_carlo=1000,quantiles=np.arange(0.05,0.96,0.05),random_seed=500):

    

    np.random.seed(random_seed)

    name = da.name

    null_hypoth_dataset = xarray.DataArray(np.zeros([len(quantiles),da[sensor_dims[0]].size,da[sensor_dims[1]].size]),
                                           coords=[( 'quantiles',quantiles),
                                                 ( sensor_dims[0],da[sensor_dims[0]].values ), 
                                                 ( sensor_dims[1],da[sensor_dims[1]].values )])


    monte_carlo_index = pandas.MultiIndex.from_product((np.arange(0,N_monte_carlo,1),np.arange(0,N_samples,1)),names=('monte_carlo_trial','sample'))
    
    
    random_idx = np.random.randint(low=0,high=da[sample_dim].size-1,size=[N_monte_carlo,N_samples]).reshape([N_monte_carlo*N_samples])



    random_composite = da.isel({sample_dim:random_idx}) #.to_dataset()
    random_composite = random_composite.to_dataset('random_composite') if name is None else random_composite.to_dataset(name='random_composite__' + name)

    
    random_composite = random_composite.assign({sample_dim:monte_carlo_index}).unstack(sample_dim).transpose('monte_carlo_trial','sample',sensor_dims[0],sensor_dims[1]).chunk({'monte_carlo_trial':-1})
    random_composite = random_composite.mean(dim='sample')
    random_quantile  = random_composite.quantile(dim='monte_carlo_trial',q=quantiles).persist()
    
    return random_quantile


def phase_composite_statistical_significance(da_to_composite,phase_partitions,N_samples,
                                    sensor_dims=['lat', 'lon'],sample_dim='time',N_monte_carlo=1000,quantiles=np.arange(0.05,0.96,0.05),random_seed=500):
    
    
    name = da_to_composite.name

    random_phase_composite = xarray.DataArray(np.zeros( [ phase_partitions.size-1,quantiles.size,da_to_composite[sensor_dims[0]].size,da_to_composite[sensor_dims[1]].size ] ),
                                              coords=[('phase_angle',phase_partitions[0:-1]),('quantiles',quantiles),
                                                      (sensor_dims[0],da_to_composite[sensor_dims[0]].values), 
                                                      (sensor_dims[1],da_to_composite[sensor_dims[1]].values)] )
    
    
    random_phase_composite = random_phase_composite.to_dataset(name='phase_composite')
    random_phase_composite['N_samples'] =  xarray.DataArray(np.zeros([phase_partitions.size-1]),coords=[('phase_angle',phase_partitions[0:-1])])
    
    
    for i_phase in range(0,phase_partitions.size-1):
        N_samples_for_current_phase = N_samples[i_phase]
        
        random_phase_composite['phase_composite'][i_phase,:,:,:] = random_sampling_for_quantiles(N_samples_for_current_phase,da_to_composite,
                                                                                                 sensor_dims=[sensor_dims[0], sensor_dims[1]],sample_dim=sample_dim,
                                                                                                 N_monte_carlo=10,quantiles=quantiles,random_seed=500)['random_composite__' + name].values
        
        random_phase_composite['N_samples'][i_phase]   =  N_samples_for_current_phase
        
    if name != None:
        
        random_phase_composite = random_phase_composite.rename({'phase_composite':'phase_composite__'+name})
        random_phase_composite = random_phase_composite.rename({'N_samples':'N_samples__'+name})


    return random_phase_composite

    
    
def simple_shifted_event_composite_average(time_steps_for_composite,da_to_composite,n_shifts,shift_increments,
                                    sensor_dims=['lat', 'lon'],sample_dim='time'):

    
    n_events = time_steps_for_composite.size
    
    
    name = da_to_composite.name
    
    time_shifts = np.linspace(-n_shifts*shift_increments,n_shifts*shift_increments,2*n_shifts+1)
    
    
    event_shifted_composite = xarray.DataArray(np.zeros([2*n_shifts+1,da_to_composite[sensor_dims[0]].size,da_to_composite[sensor_dims[1]].size]),
                                              coords=[('time_shift',time_shifts),
                                                      (sensor_dims[0],da_to_composite[sensor_dims[0]].values), 
                                                      (sensor_dims[1],da_to_composite[sensor_dims[1]].values)])
    
    event_shifted_composite_std = xarray.zeros_like(event_shifted_composite)
    
    start_date = da_to_composite[sample_dim][0].values
    end_date   = da_to_composite[sample_dim][-1].values
    for i_shift in range(0,2*n_shifts+1):
        
        shifted_times = np.asarray(time_steps_for_composite) + np.timedelta64(int(time_shifts[i_shift]),'D')
        shifted_times = shifted_times[shifted_times>start_date]
        shifted_times = shifted_times[shifted_times<end_date]
        
        event_shifted_composite[i_shift,:,:]      = da_to_composite.sel({sample_dim:shifted_times},method='nearest').mean(dim=sample_dim)
        event_shifted_composite_std[i_shift,:,:]  = da_to_composite.sel({sample_dim:shifted_times},method='nearest').std(dim=sample_dim)
        
    event_shifted_composite = event_shifted_composite.rename('shifted_composite') if name is None else event_shifted_composite.rename('shifted_composite__'+name)
    
    event_shifted_composite = event_shifted_composite.to_dataset()
    if name != None:
        
        event_shifted_composite['shifted_composite_std__'+name] = event_shifted_composite_std
    else:
        event_shifted_composite['shifted_composite_std'] = event_shifted_composite_std


    event_shifted_composite['n_events_in_composite'] = n_events
    return event_shifted_composite

def event_catalogue(input_data_array,event_object,event_magnitude,event_phase,days_before=5,days_after=5):
    
    
    event_catalogue = {input_data_array.name:[],'PC_magnitude':[],'PC_phase':[],'time':[],'days':[],
                       'initiation_time':[],'termination_time':[]}
    event_counter = 0
    
    for i_event in range(0,event_object['n_events']):
        if event_object['event_duration'][i_event]<180:
            found_overlapping_data = False

            event_start_time       = event_object['initiation_time'][i_event] -np.timedelta64(days_before,'D')
            event_termination_time = event_object['termination_time'][i_event]+np.timedelta64(days_after,'D')
           
            phase_for_event     = event_phase.sel(time=slice(event_start_time,
                                                                     event_termination_time) )
            magnitude_for_event = event_magnitude.sel(time=slice(event_start_time,
                                                                         event_termination_time))                                  
            time_to_interp = pandas.date_range(start=event_start_time,end=event_termination_time,freq='1D')

            current_data = input_data_array.squeeze().sortby(input_data_array['time'])
            test_slice = current_data.sel(time=slice(event_start_time,event_termination_time) )
            
            if test_slice.size != 0 and not np.all(np.isnan(test_slice.values)):
                         
                data_var_for_event = current_data.squeeze().interp(
                                     time=time_to_interp,method='nearest',
                                     kwargs={'fill_value':np.nan})
                
                event_catalogue[input_data_array.name].append(data_var_for_event.values)
                found_overlapping_data = True
            #END if   test_slice.size != 0  
            if found_overlapping_data:
                #print('Yay!')
                event_counter=event_counter+1
                event_catalogue['initiation_time'].append(event_start_time) 
                event_catalogue['termination_time'].append(event_termination_time) 

                
                    #event_catalogue['Sat_OLR'].append(sat_OLR_for_event.interp(time=time_to_interp,kwargs={'fill_value':np.nan}).values)
                event_catalogue['PC_magnitude'].append(magnitude_for_event.interp(time=time_to_interp,
                                                                                                     kwargs={'fill_value':np.nan}).values)
                event_catalogue['PC_phase'].append(phase_for_event.interp(time=time_to_interp,
                                                                                             kwargs={'fill_value':np.nan}).values)
                event_catalogue['time'].append(time_to_interp)
                found_overlapping_data = False
            #END if found_overlapping_data
    event_catalogue['n_events'] = event_counter
    return event_catalogue