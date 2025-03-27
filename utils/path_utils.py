import datetime
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, splev, splprep
from scipy.signal import savgol_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import thin
from sklearn.cluster import DBSCAN


def find_longest_path(binary_img):
    # Find all non-zero pixels
    y, x = np.where(binary_img > 0)
    points = list(zip(y, x))

    if len(points) == 0:
        return np.array([])

    # Create a graph where each pixel is connected to its neighbors
    n = len(points)
    adj_matrix = np.zeros((n, n))
    for i, (y1, x1) in enumerate(points):
        for j, (y2, x2) in enumerate(points):
            if abs(y1 - y2) <= 1 and abs(x1 - x2) <= 1:
                adj_matrix[i, j] = 1

    # Find the shortest path between all pairs of points
    graph = csr_matrix(adj_matrix)
    dist_matrix, predecessors = shortest_path(
        csgraph=graph, directed=False, return_predecessors=True
    )

    # Find the pair of points with the longest shortest path
    max_dist = 0
    start, end = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] > max_dist and dist_matrix[i, j] != np.inf:
                max_dist = dist_matrix[i, j]
                start, end = i, j

    # Reconstruct the path
    path = []
    i = end
    while (
        i != start and i != -9999
    ):  # -9999 is the default value when there's no predecessor
        path.append(points[i])
        i = predecessors[start, i]
    path.append(points[start])

    return np.array(path[::-1])  # Reverse the path to start from 'start'


def smooth_path(path, smoothing_factor=0.5, num_points=1000):
    if len(path) < 4:
        return path  # Not enough points to smooth

    # Separate x and y coordinates
    x, y = path.T

    # Fit a spline to the path
    tck, u = splprep([x, y], s=smoothing_factor, k=3)

    # Generate new points along the smoothed path
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck)

    # Apply additional smoothing using Savitzky-Golay filter
    window_length = min(len(x_smooth) // 10 * 2 + 1, 51)  # Must be odd
    x_smooth = savgol_filter(x_smooth, window_length, 3)
    y_smooth = savgol_filter(y_smooth, window_length, 3)

    # Combine x and y coordinates
    smoothed_path = np.column_stack((x_smooth, y_smooth))

    return smoothed_path


def adaptive_smooth_path(path, base_smoothing_factor=0.1, num_points=1000):
    if len(path) <= 4:
        return path  # Not enough points to smooth

    # Separate x and y coordinates
    x, y = path.T

    # Calculate path length
    path_length = len(path)

    # Create adaptive smoothing factor
    smoothing_factors = np.ones(path_length) * base_smoothing_factor
    edge_region = (
        path_length // 5
    )  # Adjust this value to control the size of the edge region
    smoothing_factors[:edge_region] = np.linspace(
        base_smoothing_factor * 5, base_smoothing_factor, edge_region
    )
    smoothing_factors[-edge_region:] = np.linspace(
        base_smoothing_factor, base_smoothing_factor * 5, edge_region
    )

    # Apply adaptive smoothing
    tck, u = splprep([x, y], s=0, k=3)
    u_new = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(u_new, tck)

    # Apply additional smoothing using Savitzky-Golay filter with adaptive window
    window_length = min(len(x_smooth) // 10 * 2 + 1, 51)  # Must be odd
    poly_order = 3
    x_smooth = savgol_filter(
        x_smooth, window_length, poly_order, mode="mirror"
    )
    y_smooth = savgol_filter(
        y_smooth, window_length, poly_order, mode="mirror"
    )

    # Combine x and y coordinates
    smoothed_path = np.column_stack((x_smooth, y_smooth))

    return smoothed_path


def remove_artifacts(path, window_size=7, angle_threshold=20):
    cleaned_path = []
    for i in range(len(path)):
        if i < window_size or i >= len(path) - window_size:
            cleaned_path.append(path[i])
            continue

        window = path[i - window_size : i + window_size + 1]
        main_direction = window[-1] - window[0]
        main_angle = np.arctan2(main_direction[1], main_direction[0])

        point_direction = path[i + 1] - path[i - 1]
        point_angle = np.arctan2(point_direction[1], point_direction[0])

        if abs(main_angle - point_angle) < np.radians(angle_threshold):
            cleaned_path.append(path[i])

    return np.array(cleaned_path)


def post_process_prediction(binary_img, smoothing_factor=0.9):
    if not isinstance(binary_img, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not np.array_equal(binary_img, binary_img.astype(bool)):
        raise ValueError(
            "Input image must be binary (containing only 0 and 1 or True and False)"
        )

    # Thin the binary image
    thinned = thin(binary_img)

    # Find the longest path
    path = find_longest_path(thinned)

    # Smooth the path
    smoothed_path = adaptive_smooth_path(path, smoothing_factor)

    # Remove artifacts
    cleaned_path = remove_artifacts(smoothed_path)

    # Create the output image
    result = np.zeros_like(binary_img, dtype=np.uint8)
    for x, y in cleaned_path.astype(int):
        if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
            result[x, y] = 1

    return result, cleaned_path


def extract_coordinates(binary_image):
    return np.argwhere(binary_image == 1)


def calculate_heading(start, end):
    delta_y = end[0] - start[0]
    delta_x = end[1] - start[1]
    theta = np.arctan2(delta_y, delta_x)
    return theta


def convert_to_real_world(value, scale=0.01):
    return value * scale


def theta_to_quaternion(theta):
    """Convert a heading (theta) to a quaternion representation"""
    qx = 0
    qy = 0
    qz = np.sin(theta / 2)
    qw = np.cos(theta / 2)
    return qx, qy, qz, qw


def current_timestamp():
    """Return the current timestamp in seconds since the Unix epoch"""
    return datetime.datetime.now().timestamp()


def extract_path_data(start_image, end_image, binary_image):
    start_point = extract_coordinates(start_image)[0]
    end_point = extract_coordinates(end_image)[0]
    path_points = extract_coordinates(binary_image)

    path_points = sorted(
        path_points, key=lambda point: np.linalg.norm(point - start_point)
    )

    path_data = []
    scale = 0.01
    time_per_pixel = 1  # 1 second per pixel traversal

    for i in range(len(path_points) - 1):
        start_point = path_points[i]
        # theta = calculate_heading(start_point, end_point)
        start_real_world = convert_to_real_world(start_point, scale)
        # start_quaternion = theta_to_quaternion(theta)

        timestamp = i * time_per_pixel

        path_data.append(
            {
                "timestamp": timestamp,
                "pose": {
                    "x": float(start_real_world[1]),
                    "y": float(start_real_world[0]),
                },  # , "theta": float(theta), "quaternion": start_quaternion},
            }
        )
    # Add the last point
    end_point = path_points[-1]
    end_real_world = convert_to_real_world(end_point, scale)
    # end_quaternion = theta_to_quaternion(theta)
    end_timestamp = (len(path_points) - 1) * time_per_pixel

    path_data.append(
        {
            "timestamp": end_timestamp,
            "pose": {
                "x": float(end_real_world[1]),
                "y": float(end_real_world[0]),
            },  # , "theta": float(theta), "quaternion": end_quaternion}
        }
    )
    return path_data


def interpolate_path_data(path_data, required_timestamps):
    original_timestamps = [data["timestamp"] for data in path_data]
    xs = [data["pose"]["x"] for data in path_data]
    ys = [data["pose"]["y"] for data in path_data]
    thetas = [data["pose"]["theta"] for data in path_data]

    interpolate_x = interp1d(
        original_timestamps, xs, kind="linear", fill_value="extrapolate"
    )
    interpolate_y = interp1d(
        original_timestamps, ys, kind="linear", fill_value="extrapolate"
    )
    interpolate_theta = interp1d(
        original_timestamps, thetas, kind="linear", fill_value="extrapolate"
    )

    interpolated_data = []
    for i, t in enumerate(required_timestamps):
        x = interpolate_x(t)
        y = interpolate_y(t)
        theta = interpolate_theta(t)
        quaternion = theta_to_quaternion(theta)
        interpolated_data.append(
            {
                "timestamp": i,
                "pose": {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "quaternion": quaternion,
                },
            }
        )

    return interpolated_data


def simplify_path(path_points, epsilon=1.5):
    approx_path = cv2.approxPolyDP(path_points, epsilon, closed=False)
    return approx_path[:, 0, :]


def generate_goal_poses(path_points):
    poses = []
    for i in range(len(path_points) - 1):
        x, y = path_points[i]["pose"]["x"], path_points[i]["pose"]["y"]
        x_next, y_next = (
            path_points[i + 1]["pose"]["x"],
            path_points[i + 1]["pose"]["x"],
        )
        orientation = np.arctan2(y_next - y, x_next - x)
        poses.append(
            {
                "timestamp": path_points[i]["timestamp"],
                "x": x,
                "y": y,
                "orientation": orientation,
            }
        )
    return poses


def show_path_from_arr(arr, title="Path"):
    x = [data[0] for data in arr]
    y = [data[1] for data in arr]

    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color="red")
    plt.show()


def resample_path(path, num_points):
    # Get the x and y coordinates from the path
    x = path[:, 0]
    y = path[:, 1]

    # Parameterize the path with respect to a normalized variable t (0 to 1)
    t = np.linspace(0, 1, len(path))

    # Perform spline interpolation for the path (Cubic spline)
    # s=0 ensures no smoothing of the points (pure interpolation)
    tck, u = splprep([x, y], s=100)

    # Create a new set of num_points points equally spaced along the parameter t
    t_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(t_new, tck)

    # Return the new interpolated path
    return np.vstack((x_new, y_new)).T


def smooth_path_extraction(binary_image, scale=0.01, time_per_point=1):
    # Use the post_process_prediction function to get the smoothed path
    smoothed_binary, cleaned_path = post_process_prediction(binary_image)

    path_data = []
    for i, (y, x) in enumerate(cleaned_path):
        # Convert to real-world coordinates
        real_world_x = x * scale
        real_world_y = y * scale

        timestamp = i * time_per_point

        path_data.append(
            {
                "timestamp": timestamp,
                "pose": {"x": float(real_world_x), "y": float(real_world_y)},
            }
        )

    return path_data


def convert_image_to_path(
    trajectory_image: np.ndarray,
    start_point_image: np.ndarray,
    end_point_image: np.ndarray,
    scale: float = 0.01,
    verbose=0,
    pred=False,
):
    if pred:
        thinned_trajectory_path = smooth_path_extraction(trajectory_image)
        trajectory_path_arr = np.array(
            [
                np.array([data["pose"]["x"], data["pose"]["y"]])
                for data in thinned_trajectory_path
            ]
        )

        return trajectory_path_arr

    else:
        # 1. Thinning
        thinned_trajectory_image: np.ndarray = thin(trajectory_image)
        thinned_trajectory_image: np.ndarray = thinned_trajectory_image.astype(
            int
        )

        # 2. Convert to path (timestamp, x, y, theta) -> no post-processing
        trajectory_path: List[dict] = generate_goal_poses(
            extract_path_data(
                start_point_image, end_point_image, thinned_trajectory_image
            )
        )
        trajectory_path_arr = np.array(
            [np.array([data["x"], data["y"]]) for data in trajectory_path]
        )

        if len(trajectory_path_arr) < 1:
            return None

        # 3. Cluster and find largest cluser
        cluster = DBSCAN(eps=0.2, min_samples=10)
        cluster.fit(trajectory_path_arr)

        labels = cluster.labels_
        idx, counts = np.unique(labels, return_counts=True)
        max_label = idx[np.argmax(counts)]

        if verbose:
            print(labels, max_label)

            plt.scatter(
                trajectory_path_arr[:, 0],
                trajectory_path_arr[:, 1],
                c=labels,
                marker="o",
                picker=True,
            )
            plt.xlabel("Axis X[0]")
            plt.ylabel("Axis X[1]")
            plt.show()

        trajectory_largest_cluster = trajectory_path_arr[
            np.where(labels == max_label)
        ]

        if verbose:
            show_path_from_arr(trajectory_largest_cluster)

        # 4. Post-process path to remove noise and capture overall shape
        # trajectory_downsampled = simplify_path(np.asarray(trajectory_largest_cluster/scale, dtype=np.int32))

        return trajectory_largest_cluster
