def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter)
    best_reports = []
    hypotheses = []
    probabilities = []

    for cluster_tracks, cluster_reports in clusters:
        # Generate hypotheses for each cluster
        cluster_hypotheses = []
        cluster_probabilities = []
        for track in cluster_tracks:
            for report in cluster_reports:
                # Calculate the probability of the hypothesis
                cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
                residual = np.array(report) - np.array(track)
                probability = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
                cluster_hypotheses.append((track, report))
                cluster_probabilities.append(probability)

        # Normalize probabilities
        total_probability = sum(cluster_probabilities)
        cluster_probabilities = [p / total_probability for p in cluster_probabilities]

        # Select the best hypothesis based on the highest probability
        best_hypothesis_index = np.argmax(cluster_probabilities)
        best_track, best_report = cluster_hypotheses[best_hypothesis_index]

        best_reports.append((best_track, best_report))
        hypotheses.append(cluster_hypotheses)
        probabilities.append(cluster_probabilities)

    # Log clusters, hypotheses, and probabilities
    print("JPDA Clusters:", clusters)
    print("JPDA Hypotheses:", hypotheses)
    print("JPDA Probabilities:", probabilities)
    print("JPDA Best Reports:", best_reports)

    return clusters, best_reports, hypotheses, probabilities


def perform_munkres(tracks, reports, kalman_filter):
    cost_matrix = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    for track in tracks:
        track_costs = []
        for report in reports:
            distance = mahalanobis_distance(track, report, cov_inv)
            track_costs.append(distance)
        cost_matrix.append(track_costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_reports = [(row, reports[col]) for row, col in zip(row_ind, col_ind)]

    # Log cost matrix and assignments
    print("Munkres Cost Matrix:", cost_matrix)
    print("Munkres Assignments:", list(zip(row_ind, col_ind)))
    print("Munkres Best Reports:", best_reports)

    return best_reports


def main(input_file, track_mode, filter_option, association_type):
    # ... existing code ...

    for group_idx, group in enumerate(measurement_groups):
        # ... existing code ...

        if len(group) == 1:  # Single measurement
            # ... existing code ...
        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            if association_method == 'JPDA':
                clusters, best_reports, hypotheses, probabilities = perform_jpda(
                    [track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter
                )
            elif association_method == 'Munkres':
                best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            # Compare best reports from both methods
            print(f"Comparison for Group {group_idx + 1}:")
            print("Best Reports from JPDA:", best_reports if association_method == 'JPDA' else "N/A")
            print("Best Reports from Munkres:", best_reports if association_method == 'Munkres' else "N/A")

            for track_id, best_report in best_reports:
                # ... existing code ...

    # ... existing code ...

    return tracks
