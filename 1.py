import csv
import numpy as np

def process_data(self):
    input_file = getattr(self, "input_file", None)
    track_mode = self.track_mode_combo.currentText()
    association_type = "JPDA" if self.jpda_radio.isChecked() else "Munkres"
    filter_option = self.filter_mode

    if not input_file:
        print("Please select an input file.")
        return

    print(
        f"Processing with:\nInput File: {input_file}\nTrack Mode: {track_mode}\nFilter Option: {filter_option}\nAssociation Type: {association_type}"
    )

    self.tracks = main(
        input_file, track_mode, filter_option, association_type
    )  # Process data with selected parameters

    if self.tracks is None:
        
        print("No tracks were generated.")
    else:
        print(f"Number of tracks: {len(self.tracks)}")

        # Update the plot after processing
        self.update_plot()

def main(input_file, track_mode, filter_option, association_type):
    log_file_path = 'detailed_log.csv'

    # Initialize CSV log file
    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Measurement X', 'Measurement Y', 'Measurement Z', 'Current State',
                      'Correlation Output', 'Associated Track ID', 'Associated Position X',
                      'Associated Position Y', 'Associated Position Z', 'Association Type',
                      'Clusters Formed', 'Hypotheses Generated', 'Probability of Hypothesis',
                      'Best Report Selected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    measurements = read_measurements_from_csv(input_file)

    if filter_option == "CV":
        kalman_filter = CVFilter()
    elif filter_option == "CA":
        kalman_filter = CAFilter()
    else:
        raise ValueError("Invalid filter option selected.")

    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = select_initiation_mode(track_mode)
    association_method = association_type  # 'JPDA' or 'Munkres'

    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    state_transition_times = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    last_check_time = 0
    check_interval = 0.0005  # 0.5 ms

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        current_time = group[0][3]  # Assuming the time is at index 3 of each measurement

        # Periodic checking
        if current_time - last_check_time >= check_interval:
            tracks_to_remove = check_track_timeout(tracks, current_time)
            for track_id in reversed(tracks_to_remove):
                print(f"Removing track {track_id} due to timeout")
                del tracks[track_id]
                track_id_list[track_id]['state'] = 'free'
                if track_id in firm_ids:
                    firm_ids.remove(track_id)
                if track_id in state_map:
                    del state_map[track_id]
                if track_id in hit_counts:
                    del hit_counts[track_id]
                if track_id in miss_counts:
                    del miss_counts[track_id]
            last_check_time = current_time

        if len(group) == 1:  # Single measurement
            measurement = group[0]
            assigned = False
            for track_id, track in enumerate(tracks):
                if correlation_check(track, measurement, doppler_threshold, range_threshold):
                    current_state = state_map.get(track_id, None)
                    if current_state == 'Poss1':
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                    elif current_state == 'Tentative1':
                        last_measurement = track['measurements'][-1][0]
                        dt = measurement[3] - last_measurement[3]
                        vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                        vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                        vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                    elif current_state == 'Firm':
                        kalman_filter.predict_step(measurement[3])
                        kalman_filter.update_step(np.array(sph2cart(*measurement[:3])).reshape(3, 1))

                    track['measurements'].append((measurement, current_state))
                    track['Sf'].append(kalman_filter.Sf.copy())
                    track['Sp'].append(kalman_filter.Sp.copy())
                    track['Pp'].append(kalman_filter.Pp.copy())
                    track['Pf'].append(kalman_filter.Pf.copy())
                    hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
                    assigned = True

                    # Log data to CSV
                    log_data = {
                        'Time': measurement[3],
                        'Measurement X': measurement[5],
                        'Measurement Y': measurement[6],
                        'Measurement Z': measurement[7],
                        'Current State': current_state,
                        'Correlation Output': 'Yes',
                        'Associated Track ID': track_id,
                        'Associated Position X': track['Sf'][-1][0, 0],
                        'Associated Position Y': track['Sf'][-1][1, 0],
                        'Associated Position Z': track['Sf'][-1][2, 0],
                        'Association Type': 'Single',
                        'Clusters Formed': '',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)
                    break

            if not assigned:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'

                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, 'Poss1')],
                    'current_state': 'Poss1',
                    'Sf': [kalman_filter.Sf.copy()],
                    'Sp': [kalman_filter.Sp.copy()],
                    'Pp': [kalman_filter.Pp.copy()],
                    'Pf': [kalman_filter.Pf.copy()]
                })
                state_map[new_track_id] = 'Poss1'
                state_transition_times[new_track_id] = {'Poss1': current_time}
                hit_counts[new_track_id] = 1
                initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

                # Log data to CSV
                log_data = {
                    'Time': measurement[3],
                    'Measurement X': measurement[5],
                    'Measurement Y': measurement[6],
                    'Measurement Z': measurement[7],
                    'Current State': 'Poss1',
                    'Correlation Output': 'No',
                    'Associated Track ID': new_track_id,
                    'Associated Position X': '',
                    'Associated Position Y': '',
                    'Associated Position Z': '',
                    'Association Type': 'New',
                    'Clusters Formed': '',
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': ''
                }
                log_to_csv(log_file_path, log_data)

        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            if association_method == 'JPDA':
                clusters, best_reports, hypotheses, probabilities = perform_jpda(
                    [track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter
                )
            elif association_method == 'Munkres':
                best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            for track_id, best_report in best_reports:
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *best_report, 0, 0, 0, group[0][3])
                elif current_state == 'Tentative1':
                    last_measurement = tracks[track_id]['measurements'][-1][0]
                    dt = group[0][3] - last_measurement[3]
                    vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(group[0][3])
                    kalman_filter.update_step(np.array(best_report).reshape(3, 1))

                tracks[track_id]['measurements'].append((cart2sph(*best_report) + (group[0][3], group[0][4]), current_state))
                tracks[track_id]['Sf'].append(kalman_filter.Sf.copy())
                tracks[track_id]['Sp'].append(kalman_filter.Sp.copy())
                tracks[track_id]['Pp'].append(kalman_filter.Pp.copy())
                tracks[track_id]['Pf'].append(kalman_filter.Pf.copy())
                hit_counts[track_id] = hit_counts.get(track_id, 0) + 1

                # Log data to CSV
                log_data = {
                    'Time': group[0][3],
                    'Measurement X': best_report[0],
                    'Measurement Y': best_report[1],
                    'Measurement Z': best_report[2],
                    'Current State': current_state,
                    'Correlation Output': 'Yes',
                    'Associated Track ID': track_id,
                    'Associated Position X': tracks[track_id]['Sf'][-1][0, 0],
                    'Associated Position Y': tracks[track_id]['Sf'][-1][1, 0],
                    'Associated Position Z': tracks[track_id]['Sf'][-1][2, 0],
                    'Association Type': association_method,
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': best_report
                }
                log_to_csv(log_file_path, log_data)

            # Handle unassigned measurements
            assigned_reports = set(best_report for _, best_report in best_reports)
            for report in reports:
                if tuple(report) not in assigned_reports:
                    new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                    if new_track_id is None:
                        new_track_id = len(track_id_list)
                        track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                    else:
                        track_id_list[new_track_id]['state'] = 'occupied'

                    tracks.append({
                        'track_id': new_track_id,
                        'measurements': [(cart2sph(*report) + (group[0][3], group[0][4]), 'Poss1')],
                        'current_state': 'Poss1',
                        'Sf': [kalman_filter.Sf.copy()],
                        'Sp': [kalman_filter.Sp.copy()],
                        'Pp': [kalman_filter.Pp.copy()],
                        'Pf': [kalman_filter.Pf.copy()]
                    })
                    state_map[new_track_id] = 'Poss1'
                    state_transition_times[new_track_id] = {'Poss1': current_time}
                    hit_counts[new_track_id] = 1
                    initialize_filter_state(kalman_filter, *report, 0, 0, 0, group[0][3])

                    # Log data to CSV
                    log_data = {
                        'Time': group[0][3],
                        'Measurement X': report[0],
                        'Measurement Y': report[1],
                        'Measurement Z': report[2],
                        'Current State': 'Poss1',
                        'Correlation Output': 'No',
                        'Associated Track ID': new_track_id,
                        'Associated Position X': '',
                        'Associated Position Y': '',
                        'Associated Position Z': '',
                        'Association Type': 'New',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)

        # Update states based on hit counts
        for track_id, track in enumerate(tracks):
            current_state = state_map[track_id]
            current_state_index = progression_states.index(current_state)
            if hit_counts[track_id] >= firm_threshold and current_state != 'Firm':
                state_map[track_id] = 'Firm'
                firm_ids.add(track_id)
                state_transition_times.setdefault(track_id, {})['Firm'] = current_time
            elif current_state_index < len(progression_states) - 1:
                next_state = progression_states[current_state_index + 1]
                if hit_counts[track_id] >= current_state_index + 1 and state_map[track_id] != next_state:
                    state_map[track_id] = next_state
                    state_transition_times.setdefault(track_id, {})[next_state] = current_time
            track['current_state'] = state_map[track_id]

    # Prepare data for CSV
    csv_data = []
    for track_id, track in enumerate(tracks):
        print(f"Track {track_id}:")
        print(f"  Current State: {track['current_state']}")
        print(f"  State Transition Times:")
        for state, time in state_transition_times.get(track_id, {}).items():
            print(f"    {state}: {time}")
        print("  Measurement History:")
        for state in progression_states:
            measurements = [m for m, s in track['measurements'] if s == state][:3]
            print(f"    {state}: {measurements}")
        print(f"  Track Status: {track_id_list[track_id]['state']}")
        print(f"  SF: {track['Sf']}")
        print(f"  SP: {track['Sp']}")
        print(f"  PF: {track['Pf']}")
        print(f"  PP: {track['Pp']}")
        print()

        # Prepare data for CSV
        csv_data.append({
            'Track ID': track_id,
            'Current State': track['current_state'],
            'Poss1 Time': state_transition_times.get(track_id, {}).get('Poss1', ''),
            'Tentative1 Time': state_transition_times.get(track_id, {}).get('Tentative1', ''),
            'Firm Time': state_transition_times.get(track_id, {}).get('Firm', ''),
            'Poss1 Measurements': str([m for m, s in track['measurements'] if s == 'Poss1'][:3]),
            'Tentative1 Measurements': str([m for m, s in track['measurements'] if s == 'Tentative1'][:3]),
            'Firm Measurements': str([m for m, s in track['measurements'] if s == 'Firm'][:3]),
            'Track Status': track_id_list[track_id]['state'],
            'SF': [sf.tolist() for sf in track['Sf']],
            'SP': [sp.tolist() for sp in track['Sp']],
            'PF': [pf.tolist() for pf in track['Pf']],
            'PP': [pp.tolist() for pp in track['Pp']]
        })

    # Write to CSV
    csv_file_path = 'track_summary.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Track ID', 'Current State', 'Poss1 Time', 'Tentative1 Time', 'Firm Time',
                      'Poss1 Measurements', 'Tentative1 Measurements', 'Firm Measurements',
                      'Track Status', 'SF', 'SP', 'PF', 'PP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"Track summary has been written to {csv_file_path}")

    # Add this line at the end of the function
    return tracks
