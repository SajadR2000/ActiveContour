import numpy as np
import cv2
import matplotlib.pyplot as plt


def viterbi(init_nodes, img_gradient, center_of_contour=None, gamma=50, alpha=1, m=9, compactness_factor=0.95):
    """
    Performs a modified version of viterbi algorithm for active contour algorithm
    :param init_nodes: initial points on the contour
    :param img_gradient: gradient of the image
    :param center_of_contour: center of the contour. If it's None, set as the mean of the initial nodes.
    :param gamma: coefficient of external cost
    :param alpha: coefficient of internal cost
    :param m: squared size of neighborhood
    :param compactness_factor: compactness factor for d_bar
    :return: next contour points
    """
    if center_of_contour is None:
        center_of_contour = np.mean(init_nodes, axis=0)
    # Weight of the distance from center
    center_weight = 20

    assert int(np.sqrt(m)) == np.sqrt(m)
    assert m % 2 == 1

    # print(init_nodes)
    half_window_size = int((np.sqrt(m) - 1) / 2)
    n_nodes = init_nodes.shape[0]
    # For a 3*3 neighbourhood, rows_offset = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    rows_offset = np.arange(-half_window_size, half_window_size+1).reshape((-1, 1))
    rows_offset = np.repeat(rows_offset, 2*half_window_size+1, axis=1)
    rows_offset = np.reshape(rows_offset, (-1,), 'C')
    # print(rows_offset)
    # For a 3*3 neighbourhood, cols_offset = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    cols_offset = np.arange(-half_window_size, half_window_size+1).reshape((1, -1))
    cols_offset = np.repeat(cols_offset, 2*half_window_size+1, axis=0)
    cols_offset = np.reshape(cols_offset, (-1,), 'C')
    # print(cols_offset)
    # For each node save the m locations coordinates and external cost of them
    rows = np.zeros((m, n_nodes), dtype=int)
    cols = np.zeros((m, n_nodes), dtype=int)
    grad_cost = np.zeros((m, n_nodes))
    for i in range(n_nodes):
        node = init_nodes[i, :]
        # print(node)
        window = -img_gradient[node[0]-half_window_size:node[0]+half_window_size+1,
                               node[1]-half_window_size:node[1]+half_window_size+1].copy()
        grad_cost[:, i] = np.reshape(window, (-1,), 'C')
        # print(grad_cost[:, i])
        rows[:, i] = node[0] + rows_offset
        cols[:, i] = node[1] + cols_offset
        # print(grad_cost[0, i])
        # print(grad_cost[5, i])
        # print(-img_gradient[rows[0, i], cols[0, i]])
        # print(-img_gradient[rows[5, i], cols[5, i]])
        # print(rows[:, i])
        # print(cols[:, i])
    # Check if the locations are within the image
    h, w = img_gradient.shape
    rows[rows < half_window_size] = half_window_size
    cols[cols < half_window_size] = half_window_size
    rows[rows > h - half_window_size - 1] = h - half_window_size - 1
    cols[cols > w - half_window_size - 1] = w - half_window_size - 1
    # Scale the external cost by gamma
    grad_cost = gamma * grad_cost
    # Calculate d_bar
    dx = np.diff(init_nodes[:, 0])
    dy = np.diff(init_nodes[:, 1])
    # print(dx)
    # print(dy)
    dt = np.sqrt(np.sum(np.square(dx) + np.square(dy)))
    dt += np.sqrt((init_nodes[-1, 0] - init_nodes[0, 0]) ** 2 + (init_nodes[-1, 1] - init_nodes[0, 1]) ** 2)
    dbar = dt / n_nodes
    dbar = dbar ** 2 * compactness_factor
    # Divide all costs by overflow_scale to avoid overflow
    overflow_scale = 1000
    path_cost = {"Path": [[i] for i in range(m)],
                 "Cost": grad_cost[:, 0] / overflow_scale}
    # print(path_cost)
    for i in range(1, n_nodes):
        path_cost_temp = {"Path": [[] for _ in range(m)],  # Resetting path_cost_temp
                          "Cost": np.zeros((m,))}
        for j in range(m):
            # Calculate external cost of location j for i-th node
            cost_ext = grad_cost[j, i].copy() / overflow_scale
            # print(cost_ext)
            # Calculate internal cost of location j of i-th node and all locations of (i-1)-th node
            cost_int = np.square(rows[:, i - 1] - rows[j, i]) + np.square(cols[:, i - 1] - cols[j, i])
            # Scale the internal cost by alpha
            cost_int = np.square(cost_int - dbar) / overflow_scale * alpha
            # Calculate the distance from the center
            cost_center = np.square(rows[j, i] - center_of_contour[0]) +\
                          np.square(cols[j, i] - center_of_contour[1])
            # Scale the center cost by center_weight
            cost_center = cost_center / overflow_scale * center_weight
            # print(cost_int)
            # Calculate the sum of defined costs with the previous costs
            cost_total = path_cost["Cost"] + cost_int + cost_ext + cost_center
            # print(cost_total)
            # print(cost_total.shape)
            # Choose the best location
            best = np.argmin(cost_total)
            # print(best)
            # Update the path and cost
            path_cost_temp["Path"][j] = path_cost["Path"][best] + [j]
            path_cost_temp["Cost"][j] = cost_total[best]
            # print(path_cost_temp)

        path_cost = path_cost_temp
    # print(path_cost)
    # Close the contour
    for i in range(m):
        path_i = path_cost["Path"][i]
        path_st = path_i[0]
        path_end = path_i[n_nodes - 1]
        cost_i = path_cost["Cost"][i].copy()
        cost_temp = np.square(rows[path_st, 0] - rows[path_end, n_nodes-1]) +\
                  np.square(cols[path_st, 0] - cols[path_end, n_nodes-1])
        cost_i += np.square(cost_temp - dbar) / overflow_scale
        path_cost["Cost"][i] = cost_i.copy()
    best_path = path_cost["Path"][np.argmin(path_cost["Cost"])]
    # Store the next points in an array and return them
    next_points = np.zeros(init_nodes.shape, dtype=int)
    for i in range(n_nodes):
        next_points[i, 0] = rows[best_path[i], i]
        next_points[i, 1] = cols[best_path[i], i]

    return next_points


def event_handler(event, x, y, flags, param):
    """
    Event handler. Refer to q4.py
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([y, x])
        print(y, x)
    else:
        pass


def get_user_points(input_image):
    """
    Gets the user initial contour. Refer to q4
    :param input_image: input image
    :return: user input points
    """
    user_points = []
    cv2.namedWindow("Tasbih", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Tasbih", event_handler, param=user_points)
    while True:
        cv2.imshow("Tasbih", input_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    user_points = np.array(user_points)
    return user_points


def add_contour(img, points, idx=-1, directory=None):
    """
    Draws a contour on the image and saves it in the given directory
    :param img: input image
    :param points: points on the contour
    :param idx: index of the image. This is for saving the video by ffmpeg
    :param directory: directory to save the image with the contour drawn on it
    :return: None
    """
    if directory is None:
        directory = './active_contour/contour_' + '{:03d}'.format(idx) + '.jpg'
    points_ = points.copy()
    points_[:, 0] = points[:, 1].copy()
    points_[:, 1] = points[:, 0].copy()
    img_ = img.copy()
    # Add the first point to the end so that the contour gets closed
    points_ = np.concatenate((points_, points_[0, :].reshape(1, 2)), axis=0)
    # Add the contour to the image
    img_ = cv2.drawContours(img_, [points_], 0, (255, 255, 255), thickness=2)
    # Save the output
    cv2.imwrite(directory, img_)
    return None


def points_file_reader(file_name):
    """
    This function takes the name of the file containing coordinates of the points and returns the points as an array.
    :param file_name: name of the file containing coordinates of the points
    :return: a numeric array containing coordinates of the points
    """
    with open(file_name, 'r') as f:
        points_str = f.readlines()  # Reads all the lines at once
    # The first line is the number of points:
    n_points = points_str[0]
    # Remove the next line character
    n_points = int(n_points[:-1])
    # Separate coordinates by space and assign store them in a numpy array with shape = (n_points, dim)
    dim = len(points_str[2].split(' '))
    my_points = np.zeros((n_points, dim), dtype=int)
    points_str = points_str[1:]
    for i in range(n_points):
        point_i = points_str[i].split(' ')
        for j in range(dim):
            my_points[i, j] = float(point_i[j])

    return my_points


def run(img_dir='./tasbih.jpg', max_iter=1000, tolerance=0.0001):
    """
    Runs the whole program
    :param img_dir: directory of tasbih(!)
    :param max_iter: maximum number of iterations
    :param tolerance: if the difference between two consecutive iteration is less than tolerance stop!
    :return: None
    """
    img = cv2.imread(img_dir)
    # plt.imshow(img)
    # plt.show()
    user_init_points = get_user_points(img)
    # user_init_points = points_file_reader('./test_points.txt')
    contour_expected_center = np.mean(user_init_points, axis=0)
    add_contour(img, user_init_points, 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the gradient and threshold it as explained in the report
    grad_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    grad_filter_y = grad_filter_x.T
    grad_x = cv2.filter2D(gray_img, cv2.CV_64F, grad_filter_x)
    grad_y = cv2.filter2D(gray_img, cv2.CV_64F, grad_filter_y)
    grad = np.square(grad_x) + np.square(grad_y)
    grad = grad.astype(np.float64)
    grad = grad / grad.max()
    grad_condition = grad < 0.01
    grad[grad_condition] = 0
    grad[~grad_condition] = 1
    grad = cv2.medianBlur(grad.astype(np.uint8), 7)
    # plt.imshow(grad)
    # plt.show()
    grad = grad.astype(float) * 255 ** 2

    init_points = viterbi(user_init_points, grad, center_of_contour=contour_expected_center)
    saving_imgs_frequency = 5
    for i in range(max_iter):
        next_points = viterbi(init_points, grad, center_of_contour=np.mean(init_points, axis=0))
        if i % saving_imgs_frequency == 0:
            add_contour(img, next_points, int(i / saving_imgs_frequency) + 1)
        if np.sum(np.square(next_points - init_points)) < tolerance:
            break
        init_points = next_points.copy()
    # Note that init_points = next_points
    add_contour(img, init_points, -1, './res11.jpg')


run()





