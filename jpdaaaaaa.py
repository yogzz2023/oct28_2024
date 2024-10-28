import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment
import mplcursors

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.gate_threshold = 900.21  # Chi-squared threshold

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf[:3] = np.array([[x], [y], [z]])
        self.Sf[3:] = np.array([[vx], [vy], [vz]])

    def predict_step(self, dt):
        self.Phi = np.eye(6)
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            md = float(row[11])
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((mr, ma, me, mt, md, x, y, z))
    return measurements

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def form_clusters_via_association(tracks, reports, kalman_filter):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
    chi2_threshold = kalman_filter.gate_threshold

    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            if distance < chi2_threshold:
                association_list.append((i, j))

    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]

        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]

        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    return clusters

def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter)
    best_reports = []

    for cluster_tracks, cluster_reports in clusters:
        best_track_idx, best_report = select_best_report(cluster_tracks, cluster_reports, kalman_filter)
        best_reports.append((best_track_idx, best_report))

    return clusters, best_reports

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

def plot_measurements(tracks, ax, plot_type):
    ax.clear()
    for track in tracks:
        times = [m[0][3] for m in track['measurements']]
        measurements_x = [sph2cart(*m[0][:3])[0] for m in track['measurements']]
        measurements_y = [sph2cart(*m[0][:3])[1] for m in track['measurements']]
        measurements_z = [sph2cart(*m[0][:3])[2] for m in track['measurements']]

        if plot_type == "Range vs Time":
            ax.scatter(times, measurements_x, label=f'Track {track["track_id"]} Measurement X', marker='o')
            ax.set_ylabel('X Coordinate')
        elif plot_type == "Azimuth vs Time":
            ax.plot(times, measurements_y, label=f'Track {track["track_id"]} Measurement Y', marker='o')
            ax.set_ylabel('Y Coordinate')
        elif plot_type == "Elevation vs Time":
            ax.plot(times, measurements_z, label=f'Track {track["track_id"]} Measurement Z', marker='o')
            ax.set_ylabel('Z Coordinate')

    ax.set_xlabel('Time')
    ax.set_title(f'Tracks {plot_type}')
    ax.legend()

    # Add interactive data tips
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.target.index
        track_id = tracks[index // len(tracks[0]['measurements'])]['track_id']
        measurement = tracks[index // len(tracks[0]['measurements'])]['measurements'][index % len(tracks[0]['measurements'])]
        time = measurement[0][3]
        sp = tracks[index // len(tracks[0]['measurements'])]['Sp']
        sf = tracks[index // len(tracks[0]['measurements'])]['Sf']
        plant_noise = tracks[index // len(tracks[0]['measurements'])]['Pf'][0, 0]  # Example of accessing plant noise

        sel.annotation.set(text=f"Track ID: {track_id}\nMeasurement: {measurement}\nTime: {time}\nSp: {sp}\nSf: {sf}\nPlant Noise: {plant_noise}")

# Example usage
if __name__ == "__main__":
    # Initialize filter
    kalman_filter = CVFilter()

    # Read measurements from CSV
    file_path = 'measurements.csv'  # Replace with your CSV file path
    measurements = read_measurements_from_csv(file_path)

    # Group measurements
    measurement_groups = form_measurement_groups(measurements)

    # Example tracks and reports
    tracks = [{'track_id': 0, 'measurements': [((0, 0, 0, 0), 'Poss1')]}]
    reports = [(1, 1, 1), (2, 2, 2)]

    # Perform JPDA
    clusters, best_reports = perform_jpda(tracks, reports, kalman_filter)

    # Plot results
    fig, ax = plt.subplots()
    plot_measurements(tracks, ax, "Range vs Time")
    plt.show()
