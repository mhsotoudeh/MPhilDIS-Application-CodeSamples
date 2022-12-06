import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import matplotlib.animation
import sys
import random
import datetime
import os
import json
import pyodbc

import time
alg_start = time.time()

import warnings
warnings.filterwarnings("ignore")


class ObservationBlock:
    def __init__(self, id, name, right_ascension, declination, exposure, continuous_exposure=False, t_min=None, t_max=None):
        self.id = id
        self.name = name
        self.right_ascension = right_ascension
        self.declination = np.deg2rad(declination)
        self.exposure = exposure
        self.continuous_exposure = continuous_exposure
        self.t_start_reducing_azimuth = None
        self.t_end_reducing_azimuth = None
        self.t_ascending_above_ceiling = None
        self.t_descending_below_ceiling = None
        self.t_pve_cross = None
        self.t_pvw_cross = None

        ha = calculate_ha_from_altitude(set_altitude, self.declination)
        if ha == 24:
            self.t_rise = 0
            self.t_set = 24
        else:
            self.t_rise = adjust_value(-ha + self.right_ascension)  # Local Sidereal Time
            self.t_rise = adjust_value(self.t_rise - start_time)

            self.t_set = adjust_value(ha + self.right_ascension)  # Local Sidereal Time
            self.t_set = adjust_value(self.t_set - start_time)

        self.t_transit = adjust_value(self.right_ascension - start_time)

        self.delta_t_available = 2*ha

        # Start Time and Best Start Time
        self.t_start = None
        self.t_start_best = adjust_value(self.t_transit - self.exposure/2)
        self.t_start_optimum = self.t_start_best

        if not self.fits_into_time(self.t_start_best): # If t_start_best does not fit into time
            delta_to_rise = adjust_value(self.t_start_best - self.t_rise)
            delta_to_set = adjust_value(self.t_start_best - self.t_set)

            if delta_to_rise < delta_to_set:
                self.t_start_optimum = self.t_rise
            else:
                self.t_start_optimum = self.t_set - self.exposure

        # Northern Meridian Transit
        if self.declination > latitude:
            ha = np.rad2deg(np.arccos(np.tan(latitude) / np.tan(self.declination))) / 15

            self.t_start_reducing_azimuth = adjust_value(-ha + self.right_ascension)  # Local Sidereal Time
            self.t_start_reducing_azimuth = adjust_value(self.t_start_reducing_azimuth - start_time)

            self.t_end_reducing_azimuth = adjust_value(ha + self.right_ascension)  # Local Sidereal Time
            self.t_end_reducing_azimuth = adjust_value(self.t_end_reducing_azimuth - start_time)

        # Ceiling Intersection
        if latitude - minimum_z < self.declination < latitude + minimum_z:
            ha = calculate_ha_from_altitude(np.pi/2 - minimum_z, self.declination)

            self.t_ascending_above_ceiling = adjust_value(-ha + self.right_ascension)  # Local Sidereal Time
            self.t_ascending_above_ceiling = adjust_value(self.t_ascending_above_ceiling - start_time)

            self.t_descending_below_ceiling = adjust_value(ha + self.right_ascension)  # Local Sidereal Time
            self.t_descending_below_ceiling = adjust_value(self.t_descending_below_ceiling - start_time)

        # Prime Vertical Intersection
        if 0 < self.declination < latitude:
            ha = calculate_ha_on_pv(self.declination)

            self.t_pve_cross = adjust_value(-ha + self.right_ascension)  # Local Sidereal Time
            self.t_pve_cross = adjust_value(self.t_pve_cross - start_time)

            self.t_pvw_cross = adjust_value(ha + self.right_ascension)  # Local Sidereal Time
            self.t_pvw_cross = adjust_value(self.t_pvw_cross - start_time)

    def fits_into_time(self, t_start, added_time=0):
        t_block = self.exposure + added_time

        if self.continuous_exposure is True:
            if self.t_ascending_above_ceiling < t_start < self.t_descending_below_ceiling or \
                    self.t_ascending_above_ceiling < t_start + self.exposure < self.t_descending_below_ceiling or \
                    t_start < self.t_ascending_above_ceiling < t_start + self.exposure or \
                    t_start < self.t_descending_below_ceiling < t_start + self.exposure:
                return False

        if t_start + t_block > length:
            return False

        if self.t_set > self.t_rise:
            if t_start > self.t_rise and t_start+t_block < self.t_set:
                return True
            else:
                return False
        else:
            if (t_start > self.t_rise and t_start+t_block < self.t_set+24) or (t_start > 0 and t_start+t_block < self.t_set):
                return True
            else:
                return False

    def __eq__(self, other):
        assert isinstance(other, ObservationBlock)
        return self.name == other.name

    def __lt__(self, other):
        assert isinstance(other, ObservationBlock)

        if self.t_start is None or other.t_start is None:
            return False

        if self.t_start < other.t_start:
            return True
        else:
            return False

    def __str__(self):
        return str(self.name)# + ", " + str(self.t_start)

    def __repr__(self):
        return str(self.name)# + ", " + str(self.t_start)


def adjust_value(time):
    if time < 0:
        return time + 24
    elif time > 24:
        return time - 24
    else:
        return time


def calculate_separation(dec1, ra1, dec2, ra2):
    delta_alpha = np.deg2rad((ra2-ra1)*15)
    cos_sep = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(delta_alpha)
    return np.rad2deg(np.arccos(cos_sep))


def calculate_ha_on_pv(dec):
    cos_ha = np.tan(dec) / np.tan(latitude)
    if cos_ha < -1 or cos_ha > 1:
        return 24
    else:
        return np.rad2deg(np.arccos(cos_ha)) / 15


def calculate_ha_from_altitude(altitude, dec):
    cos_ha = ( np.sin(altitude) - np.sin(dec)*np.sin(latitude) ) / ( np.cos(dec)*np.cos(latitude) )
    if cos_ha < -1 or cos_ha > 1:
        return 24
    else:
        return np.rad2deg(np.arccos(cos_ha)) / 15


def calculate_altitude(dec, ha):
    sin_a = np.sin(dec)*np.sin(latitude) + np.cos(dec)*np.cos(latitude)*np.cos(np.deg2rad(ha*15))
    return np.rad2deg(np.arcsin(sin_a))


def calculate_azimuth(dec, ha):
    A = np.rad2deg(np.arctan(np.sin(np.deg2rad(-ha*15)) / (np.cos(latitude)*np.tan(dec) - np.sin(latitude)*np.cos(np.deg2rad(-ha*15)))))
    A_prime = np.rad2deg(np.arcsin(np.cos(dec) * np.sin(np.deg2rad(-ha*15)) / np.cos(np.deg2rad(calculate_altitude(dec, ha)))))

    if (A - A_prime) < 0.001:
        if A < 0:
            A += 360
        return A
    else:
        return 180 + A


def boost_schedule(schedule, index, stepsize=0.05): #timestep):
    pointing_times = pointing_and_tracking_calculations(schedule)[0]

    current_overlaps = count_overlaps(schedule, pointing_times)
    current_merit = transit_merit(schedule, index)
    start_time = schedule[index]
    block = observation_blocks[index]
    sign = np.sign(block.t_transit - (start_time+block.exposure/2))
    new_schedule = schedule.copy()

    # while True:
    for i in range(100):
        new_start_time = start_time + sign*stepsize
        if 0 < new_start_time < length:
            new_schedule[index] = new_start_time
            # pointing_times = pointing_and_tracking_calculations(new_schedule) # Looking for more efficient way! # Current assumption: Pointing time does not change too much
            if count_overlaps(new_schedule, pointing_times) <= current_overlaps and block.fits_into_time(new_start_time) and \
                    transit_merit(new_schedule, index) >= current_merit:
                start_time = new_start_time
                # print("I'm stuck here! New start time for " + str(block) + " is: " + str(new_start_time))
            else:
                break
        else:
            break

    new_schedule[index] = start_time
    return new_schedule


def pointing_time_and_azimuth(dec_from, ra_from, dec_to, ra_to, lst_from, lst_to, A_telescope):
    # lst from is the time for which local coordinates of first star should be calculated
    # lst to is the time for which local coordinates of second star should be calculated

    ha_from = adjust_value(lst_from - ra_from)
    ha_to = adjust_value(lst_to - ra_to)

    a_from = calculate_altitude(dec_from, ha_from)
    a_to = calculate_altitude(dec_to, ha_to)
    delta_a = abs(a_to - a_from)

    A_from = A_telescope
    A_to = calculate_azimuth(dec_to, ha_to)

    if A_from > 0:
        if A_to > 270:  # Telescope mount limit
            A_to = -(360 - A_to)
        elif A_to - A_from > 180:  # Finding shortest path
            A_to = -(360 - A_to)
    else:
        if A_to > 90:  # Telescope mount limit
            A_to = -(360 - A_to)
        if A_to - A_from < -180:  # Finding shortest path
            A_to += 360

    delta_A = A_to - A_from
    t_pointing = max(abs(delta_A) / azimuth_axis_velocity, delta_a / altitude_axis_velocity)

    return t_pointing, A_to


def tracking_coordinates(A_start_of_tracking, target_block, t_start_of_tracking, t_end_of_tracking):
    assert isinstance(target_block, ObservationBlock)

    dec_target, ra_target = target_block.declination, target_block.right_ascension

    lst_start_of_tracking = adjust_value(start_time + t_start_of_tracking)
    ha_start_of_tracking = adjust_value(lst_start_of_tracking - ra_target)
    lst_end_of_tracking = adjust_value(start_time + t_end_of_tracking)
    ha_end_of_tracking = adjust_value(lst_end_of_tracking - ra_target)

    total_rotation_time = 0

    a_start_of_tracking = calculate_altitude(dec_target, ha_start_of_tracking)
    a_end_of_tracking = calculate_altitude(dec_target, ha_end_of_tracking)

    A_end_of_tracking = calculate_azimuth(dec_target, ha_end_of_tracking)
    # Note: Stars for which A is not monotonically increasing (90-delta < 90-phi) total rotation does not occur,
    # since A is always less than 90 E or 90 W. However, their azimuth may decrease over time

    if target_block.t_start_reducing_azimuth is not None and 0 < ha_end_of_tracking < 12:
        A_end_of_tracking = -(360 - A_end_of_tracking)
        return A_start_of_tracking, A_end_of_tracking, a_start_of_tracking, a_end_of_tracking, total_rotation_time

    # If telescope azimuth at start of tracking is negative
    if A_start_of_tracking < 0:
        A_end_of_tracking = -(360 - A_end_of_tracking)

    # If sign of A changes during tracking (only for circumpolar stars)
    if A_end_of_tracking < A_start_of_tracking:
        A_end_of_tracking = 360 + A_end_of_tracking

    # Total rotation correction (Only occurs if A_telescope exeeds +270 degree)
    if A_end_of_tracking > 270: # Implicitly: and if A_start_of_tracking > 0 (Only needs to be taken into account if A_start_of_tracking < 0 makes sense, eg. observing circumpolar star for more than 36 hours)
        total_rotation_time = 2.0/60
        A_start_of_tracking = -(360 - A_start_of_tracking)
        A_end_of_tracking = -(360 - A_end_of_tracking)

    return A_start_of_tracking, A_end_of_tracking, a_start_of_tracking, a_end_of_tracking, total_rotation_time


def pointing_and_tracking_calculations(schedule):
    zipped_pairs = zip(schedule, observation_blocks)
    sorted_blocks = [x for _, x in sorted(zipped_pairs)]
    sorted_schedule = schedule.copy()
    sorted_schedule.sort()

    ha_start = adjust_value( (start_time + sorted_schedule[0]) - sorted_blocks[0].right_ascension)
    A_start_of_tracking = calculate_azimuth(sorted_blocks[0].declination, ha_start)
    if A_start_of_tracking > 180:
        A_start_of_tracking = -(360 - A_start_of_tracking)

    t_start_of_tracking = sorted_schedule[0]
    t_end_of_tracking = adjust_value(sorted_schedule[0] + sorted_blocks[0].exposure)
    A_start_of_tracking, A_end_of_tracking, a_start_of_tracking, a_end_of_tracking = tracking_coordinates(A_start_of_tracking, sorted_blocks[0], t_start_of_tracking, t_end_of_tracking)[:4]

    times = []
    azimuths = [A_start_of_tracking, A_end_of_tracking]
    altitudes = [a_start_of_tracking, a_end_of_tracking]
    for i in range(len(schedule)-1):
        current_block = sorted_blocks[i]
        next_block = sorted_blocks[i+1]

        lst_start_position = adjust_value(start_time + sorted_schedule[i] + current_block.exposure)
        lst_end_position =  adjust_value(start_time + sorted_schedule[i+1])
        t_pointing, A_start_of_tracking = pointing_time_and_azimuth(current_block.declination, current_block.right_ascension,
                                  next_block.declination, next_block.right_ascension, lst_start_position, lst_end_position, A_end_of_tracking)
        times.append(t_pointing)

        # Update A_telescope
        t_start_of_tracking = sorted_schedule[i+1]
        t_end_of_tracking = adjust_value(sorted_schedule[i+1] + next_block.exposure)
        A_start_of_tracking, A_end_of_tracking, a_start_of_tracking, a_end_of_tracking, total_rotation_time = tracking_coordinates(A_start_of_tracking, next_block, t_start_of_tracking, t_end_of_tracking)
        azimuths.extend([A_start_of_tracking, A_end_of_tracking])
        altitudes.extend([a_start_of_tracking, a_end_of_tracking])

    times.append(0.0)

    ids, start_azimuths, end_azimuths, start_altitudes, end_altitudes = [], [], [], [], []
    for i in range(len(schedule)):
        ids.append(sorted_blocks[i].id)
        start_azimuths.append(azimuths[2*i])
        end_azimuths.append(azimuths[2*i + 1])
        start_altitudes.append(altitudes[2*i])
        end_altitudes.append(altitudes[2*i + 1])

    zipped_pairs = zip(ids, times)
    times = [x for _, x in sorted(zipped_pairs)]
    zipped_pairs = zip(ids, start_azimuths)
    start_azimuths = [x for _, x in sorted(zipped_pairs)]
    zipped_pairs = zip(ids, end_azimuths)
    end_azimuths = [x for _, x in sorted(zipped_pairs)]
    zipped_pairs = zip(ids, start_altitudes)
    start_altitudes = [x for _, x in sorted(zipped_pairs)]
    zipped_pairs = zip(ids, end_altitudes)
    end_altitudes = [x for _, x in sorted(zipped_pairs)]

    return times, start_azimuths, end_azimuths, start_altitudes, end_altitudes


def transit_merit(schedule, index):
    midtime = schedule[index] + observation_blocks[index].exposure/2

    t_rise = observation_blocks[index].t_rise
    t_transit = observation_blocks[index].t_transit
    t_set = observation_blocks[index].t_set

    if t_set < t_rise:
        if midtime < t_set:
            midtime += 24
        if t_transit < t_set:
            t_transit += 24
        t_set += 24

    delta_t_transit = np.abs(midtime - t_transit)

    merit = (1 - 2 * delta_t_transit / observation_blocks[index].delta_t_available)
    return merit


def calculate_transit_merits(schedule):
    merits = []
    for i in range(len(schedule)):
        merits.append(transit_merit(schedule, i))
    return merits


def descend_time_merit(schedule, index):
    midtime = schedule[index] + observation_blocks[index].exposure / 2

    t_rise = observation_blocks[index].t_rise
    t_set = observation_blocks[index].t_set

    if t_set < t_rise:
        if midtime < t_set:
            midtime += 24
        t_set += 24

    delta_t_set = t_set - midtime

    merit = (delta_t_set / observation_blocks[index].delta_t_available)
    # if merit > 0.5: # To avoid early scheduling of eastern stars
    #     merit = 1.0
    return merit


def calculate_descend_time_merits(schedule):
    merits = []
    for i in range(len(schedule)):
        merits.append(descend_time_merit(schedule, i))
    return merits


def list_corrects(schedule, added_times):
    result = []
    for i in range(len(schedule)):
        if observation_blocks[i].fits_into_time(schedule[i], added_times[i]):
            result.append(True)
        else:
            result.append(False)
    return result


def count_corrects(schedule, added_times):
    return np.sum(list_corrects(schedule, added_times))  # It is better to change "added_times" name!


def count_overlaps(schedule, added_times=None):
    if added_times is None:
        added_times = [0.0 for i in range(len(schedule))]  # It is better to change "added_times" name!

    result = 0
    for i in range(len(schedule)):
        for j in range(len(schedule)):
            if j == i:
                continue
            if schedule[j] <= schedule[i] <= schedule[j]+observation_blocks[j].exposure+added_times[j]:
                result += 1
    return result


def generate_population(n, blocks, times):
    population = []
    length = len(blocks)
    for i in range(n):
        temp = times.copy()
        random.shuffle(temp)
        population.append(temp[:length])

    return population


def calculate_delays(schedule):
    discontinuous_blocks = []
    delays = []
    for i in range(len(schedule)):
        delay = 0.0
        tracking_start = schedule[i]
        tracking_end = schedule[i]+observation_blocks[i].exposure
        t_enter = observation_blocks[i].t_ascending_above_ceiling
        t_exit = observation_blocks[i].t_descending_below_ceiling
        if t_enter is not None:
            if t_enter < tracking_start < t_exit:
                schedule[i] = t_exit
            elif tracking_start < t_enter:
                if t_enter < tracking_end < t_exit:
                    discontinuous_blocks.append(i)
                    delay = tracking_end - t_enter
                elif tracking_end > t_exit:
                    discontinuous_blocks.append(i)
                    delay = t_exit - t_enter
        delays.append(delay)

    return schedule, delays, discontinuous_blocks


def calculate_fitness(population):
    fitnesses = []
    for schedule in population:
        schedule, delays = calculate_delays(schedule)[:2]
        pointing_times = pointing_and_tracking_calculations(schedule)[0]
        added_times = [sum(x) for x in zip(pointing_times, delays)]

        fitness = 0
        fitness -= count_overlaps(schedule, added_times)*overlap_factor
        fitness += count_corrects(schedule, delays)*correctness_factor
        fitness += ( np.sum(np.power(calculate_transit_merits(schedule), transit_slope)) / len(schedule) ) * transit_factor
        fitness += ( np.sum(np.power(calculate_descend_time_merits(schedule), sink_slope)) / len(schedule) ) * sink_factor
        fitness -= np.sum(pointing_times)/length * pointing_factor

        fitnesses.append(fitness)

    return fitnesses


def select_mating_pool(n, population, fitnesses):
    zipped_pairs = zip(fitnesses, population)
    sorted_population = [x for _, x in sorted(zipped_pairs)]
    sorted_population.reverse()
    return sorted_population[:n]


def crossover(n, parents):
    offspring = []

    for i in range(0,n,2):
        crossover_point = random.randint(1, len(parents[0])-1)

        child1 = parents[i % len(parents)][:crossover_point]
        second_half = parents[(i+1) % len(parents)][crossover_point:]
        child1.extend(second_half)
        offspring.append(child1)

        child2 = parents[(i+1) % len(parents)][:crossover_point]
        second_half = parents[i % len(parents)][crossover_point:]
        child2.extend(second_half)
        offspring.append(child2)

    return offspring


def mutate(offspring):
    length = len(offspring[0])-1
    for chromosome in offspring:
        block = random.randint(0, length)
        chromosome = boost_schedule(chromosome, block)

    return offspring


test_case_id = 14
# Import Parameters
address = 'E:/Astronomy/Pro/INO340/Object Scheduling/Implementation/Test/3 - Sotoudeh/Test.xlsx'
params = pd.read_excel(address, str(test_case_id)+' - Settings', header=3)

## Scientific Parameters -> Observatory
latitude = np.deg2rad(params['Latitude'][0])
set_altitude = np.deg2rad(params['Set Altitude'][0])
altitude_axis_velocity = params['Alt. Axis Velocity'][0]*3600.0
azimuth_axis_velocity = params['Az. Axis Velocity'][0]*3600.0
minimum_z = np.deg2rad(90 - params['Maximum Altitude'][0])

## Scientific Parameters -> Time
start_solar_time = params['Start ZT'][0]
start_time = params['Start LST'][0]
end_time = params['End LST'][0]
length = adjust_value(end_time - start_time)

## Algorithm Parameters -> Genetic
timestep = params['Timestep'][0]
population_size = params['Population Size'][0]
mating_pool_size = params['Mating Pool Size'][0]
offspring_size = params['Offspring Size'][0]
num_of_generations = params['# Generations'][0]

## Algorithm Parameters -> Evaluation Coefficients
overlap_factor = params['Overlap Factor'][0]
correctness_factor = params['Correctness Factor'][0]
pointing_factor = params['Pointing Factor'][0]
transit_factor = params['Transit Factor'][0]
transit_slope = params['Transit Slope'][0]
sink_factor = params['Sink Factor'][0]
sink_slope = params['Sink Slope'][0]


# Import Data
address = 'E:/Astronomy/Pro/INO340/Object Scheduling/Implementation/Test/3 - Sotoudeh/Test.xlsx'
input = pd.read_excel(address, str(test_case_id), header=1)
num_of_blocks = len(input)
observation_blocks = []

for i in range(num_of_blocks):
    observation_blocks.append(ObservationBlock(i, input['Name'][i], input['R.A. (h)'][i], input['Dec. (deg)'][i],
                                               input['Exposure (h)'][i]))#, input['t_min'][i], input['t_max'][i]))

## Pop impossible blocks
for i in range(num_of_blocks-1, -1, -1):
    block = observation_blocks[i]
    observation_impossible = False

    available_time = adjust_value(block.t_set - block.t_rise)
    if available_time < block.exposure:
        observation_impossible = True

    elif block.t_rise < block.t_set:
        if length - block.t_rise < block.exposure:
            observation_impossible = True

    else:
        if block.t_rise < length:
            if length - block.t_rise < block.exposure:
                observation_impossible = True
            elif block.t_set < block.exposure:
                observation_impossible = True

        else:
            if block.t_set < block.exposure:
                observation_impossible = True

    if observation_impossible is True:
        print(str(observation_blocks[i].name) + " is invalid.")
        observation_blocks.pop(i)

num_of_blocks = len(observation_blocks)
if num_of_blocks == 0:
    print("All blocks are invalid.")
    exit()
if num_of_blocks == 1:
    print("Only 1 block is valid:")
    observation_blocks[0].t_start = observation_blocks[0].t_start_best
    print(str(observation_blocks[0]) + ": " + str(round(observation_blocks[0].t_start, 2)) + " / Best: " + str(round(observation_blocks[i].t_start_best, 2)))
    exit()

# Driving Algorithm
num_of_possible_times = int(length / timestep)
possible_start_times = [i*timestep for i in range(num_of_possible_times-1)]
population = generate_population(population_size, observation_blocks, possible_start_times)

fitnesses = None
for i in range(num_of_generations):
    if (i+1) % 10 == 0:
        print("Iteration #" + str(i+1))
    fitnesses = calculate_fitness(population)
    population = select_mating_pool(mating_pool_size, population, fitnesses)
    offspring = mutate(crossover(offspring_size, population))
    population.extend(offspring)


# Results
zipped_pairs = zip(fitnesses, population)
population = [x for _, x in sorted(zipped_pairs)]
population.reverse()
best_schedule = population[0]

# Order observation blocks by t_start (ascending)
zipped_pairs = zip(best_schedule, observation_blocks)
observation_blocks = [x for _, x in sorted(zipped_pairs)]
best_schedule.sort()
for i in range(len(observation_blocks)):
    observation_blocks[i].id = i

# Boost
# indices = np.arange(0, num_of_blocks)
# for j in range(100):
#     random.shuffle(indices)
for i in range(num_of_blocks):
    best_schedule = boost_schedule(best_schedule, i)

point_results = pointing_and_tracking_calculations(best_schedule)
pointing_times = point_results[0]
delays, discontinuous_blocks = calculate_delays(best_schedule)[1:]
added_times = [sum(x) for x in zip(pointing_times, delays)]

correct_states = list_corrects(best_schedule, pointing_times)
for i in range(num_of_blocks-1, -1, -1):
    if correct_states[i] is False:
        print("Improper time: ", best_schedule[i])
        best_schedule[i] = 25  # To exclude from graph

## Print
for i in range(num_of_blocks):
    observation_blocks[i].t_start = best_schedule[i]
    print(str(observation_blocks[i]) + ": " + str(round(observation_blocks[i].t_start, 2)) + " / Optimum: " + str(round(observation_blocks[i].t_start_optimum, 2)))

print("Observation Blocks Status: " + str(correct_states))
print("Number of Overlaps: " + str(count_overlaps(best_schedule)))
print("Number of Successful Observations: " + str(np.sum(correct_states)))
print("Transit Merits", np.round(calculate_transit_merits(best_schedule), 2))
print("Pointing Durations (s): ", 3600 * np.round(pointing_times, 2))
print("Delays (s): ", 3600 * np.round(delays, 2))
print("Start Azimuths: ", np.round(point_results[1], 2))
print("End Azimuths: ", np.round(point_results[2], 2))
print("Start Altitudes: ", np.round(point_results[3], 2))
print("End Altitudes: ", np.round(point_results[4], 2))


## Discontinuity Check
if len(discontinuous_blocks) == 0:
    print("No blocks have exposure discontinuity.")
else:
    names = []
    for id in discontinuous_blocks:
        names.append(observation_blocks[id].name)
    print("Discontinuous Blocks: ", names)


## Graph
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('axes', titlesize=BIGGER_SIZE)       # Chart title
plt.rc('axes', labelsize=MEDIUM_SIZE)       # x and y titles
plt.rc('xtick', labelsize=SMALL_SIZE)       # x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # y tick labels
plt.rc('legend', fontsize=8)                # Legend

plt.rc('axes', linewidth=2)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=2)

# plt.rc('figure', figsize=(8,4))         # Image size

x_label = 'Local Time (h)'
y_label = 'Altitude (degree)'
plt_title = 'Altitude vs. Time'
fig = plt.figure()
ax = fig.add_subplot(111, xlabel=x_label, ylabel=y_label, title=plt_title)

for i in range(num_of_blocks):
    # if list_corrects(sorted_population[0])[i] is not False:
    block = observation_blocks[i]
    declination = block.declination
    transit_time = block.t_transit
    if block.t_rise < block.t_set:
        t_list = np.arange(block.t_rise, block.t_set, 0.01)
        a_list = [calculate_altitude(declination, transit_time-t) for t in t_list]
    else:
        t_list1 = np.arange(block.t_rise, length, 0.01)
        t_list2 = np.arange(0, block.t_set, 0.01)
        t_list = np.concatenate((t_list1, t_list2))
        a_list = [calculate_altitude(declination, transit_time-t) for t in t_list]

    ax.plot(t_list, a_list, color='#E8E8E8', linewidth=1)

for i in range(num_of_blocks):
    # if list_corrects(sorted_population[0])[i] is not False:
    t_list = np.arange(observation_blocks[i].t_start, observation_blocks[i].t_start+observation_blocks[i].exposure, 0.01)
    declination = observation_blocks[i].declination
    transit_time = observation_blocks[i].t_transit
    a_list = [calculate_altitude(declination, transit_time-t) for t in t_list]

    ax.plot(t_list, a_list, linewidth=2, label=observation_blocks[i].name)#, color='red')

plt.tight_layout()

lgd = plt.legend(loc='lower right', bbox_to_anchor=(1, 0.84), ncol=5)

h = int(start_solar_time)
m = int((start_solar_time-h) * 60)
# m = m - m%30
a = datetime.datetime(year=2000, month=1, day=1, hour=h, minute=m)
d = datetime.timedelta(minutes=30)
labels = []
for i in range(int(2*length+1)):
    labels.append(a.strftime("%H:%M"))
    a = a + d
plt.xticks(np.arange(0, length+0.5, 0.5), labels)
plt.xlim(0, length)  # 24

plt.ylim(set_altitude, 90)
plt.grid(True, linewidth=0.5, linestyle='--', color='#EBEBEB')  # c = '.75'

destination = str(test_case_id) + ' - ' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M")) + ' ' + str(os.path.basename(__file__))[:-3] + '.png'  # %Y%m%d_%H%M%S_%f
plt.savefig(destination, dpi=1000)
plt.show()

print("Total runtime: " + str(round((time.time()-alg_start)/60, 2)) + " minutes")