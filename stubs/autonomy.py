import logging
from numbers import Real
import time
from typing import List

from torch import true_divide

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things
from tilsdk.mock_robomaster.robot import Robot                 # Use this for the simulator
# from robomaster.robot import Robot                              # Use this for real robot

# Import your code
# from cv_service_torch import CVService, MockCVService  # using pytorch
from cv_service import CVService, MockCVService  # using onnx
from nlp_service import NLPService
from planner import Planner
import cv2
import numpy as np
import random

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# Define config variables in an easily accessible location
# You may consider using a config file
REACHED_THRESHOLD_M = 0.3   # TODO: Participant may tune.
ANGLE_THRESHOLD_DEG = 20.0  # TODO: Participant may tune.
ROBOT_RADIUS_M = 0.17       # TODO: Participant may tune.
NLP_MODEL_DIR = 'data/models/nlp'          # TODO: Participant to fill in.
CV_MODEL_DIR = 'data/models/cv'           # TODO: Participant to fill in.
CAT_2_NAME = {1: 'Fallen', 0: 'Standing'}

# Convenience function to update locations of interest.
def update_locations(old:List[RealLocation], new:List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                old.append(loc)

def get_random_loi(map_) -> RealLocation:
    while True:
        # width, height = random.randint(0, map_.width / 2), random.randint(0, map_.height)
        # width, height = 60, 80
        width, height = 30, 30
        random_grid_loc = GridLocation(width, height)
        if map_.passable(random_grid_loc) and map_.in_bounds(random_grid_loc):
            break
    return map_.grid_to_real(random_grid_loc)

def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    # cv_service = MockCVService(model_dir=CV_MODEL_DIR)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)
    loc_service = LocalizationService(host='localhost', port=5566)
    # loc_service = LocalizationService(host='192.168.20.56', port=5522)
    # rep_service = ReportingService(host='192.168.20.56', port=5522)
    rep_service = ReportingService(host='localhost', port=5566)

    robot = Robot()
    robot.initialize(conn_type="sta")
    # robot.initialize(conn_type="sta", sn="3JKDH2T001U0H4")  # for real robot
    robot.camera.start_video_stream(display=False, resolution='720p')

    # Start the run
    rep_service.start_run()

    # Initialize planner
    map_:SignedDistanceGrid = loc_service.get_map()
    map_ = map_.dilated(1.5*ROBOT_RADIUS_M/map_.scale)
    # TODO: process map?
    planner = Planner(map_, sdf_weight=0.5)

    # Initialize variables
    seen_clues = set()
    curr_loi:RealLocation = None
    path:List[RealLocation] = []
    lois:List[RealLocation] = []
    curr_wp:RealLocation = None

    random_exploration_mode:bool = False

    # Initialize tracker
    # TODO: Participant to tune PID controller values. Currently all 0 so it wont move when no goal
    # https://students.iitk.ac.in/robocon/docs/doku.php?id=robocon16:programming:pid_controller#:~:text=The%20process%20of%20tuning%20is,to%20set%20ki%20or%20kd.
    # set Kp first, if need then half kp, set ki
    # if really need then set kd
    # tracker = PIDController(Kp=(0.25, 0.25), Kd=(0.0, 0.0), Ki=(0.0, 0.0))
    tracker = PIDController(Kp=(0.25, 0.25), Kd=(0.1, 0.01), Ki=(0.05, 0.05))
    # tracker = PIDController(Kp=(0.0, 0.0), Kd=(0.0, 0.0), Ki=(0.0, 0.0))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5)

    # Define filter to exclude clues seen before
    new_clues = lambda c: c.clue_id not in seen_clues

   # Main loop
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()
        pose = pose_filter.update(pose)
        print('pose', pose)
        img = robot.camera.read_cv2_image(strategy='newest')

        if not pose:
            # now new data, continue to next iteration.
            continue

        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        if clues:
            if random_exploration_mode:  # TODO: check logic
                print('********* clue ******************')

                random_exploration_mode = False
                lois = []
                curr_loi = None

            new_lois = nlp_service.locations_from_clues(clues)
            update_locations(lois, new_lois)
            seen_clues.update([c[0] for c in clues])

        # Process image and detect targets
        start = time.time()
        targets = cv_service.targets_from_image(img)
        print('CV inference time', time.time() - start)

        # Submit targets
        if targets:
            logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
            logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))

            for d in targets:
                x, y, w, h = list(map(int, d.bbox))
                cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0), 2)
                cv2.circle(img, (x, y), 1, (0, 255, 0))
                cv2.putText(img, f'{CAT_2_NAME[d.cls]}', (x+int(w/2)+10,y+int(h/2)), 0, 0.3, (0,255,0))
            cv2.imwrite(f"./data/imgs/det.jpg", img)

        if not curr_loi:
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new (check logic)
                random_exploration_mode = True
                # pick a random RealLocation to go to, and go until a clue is found
                random_loi: RealLocation = get_random_loi(map_)
                print('###################', random_loi, map_.real_to_grid(random_loi))
                time.sleep(1)
                lois.append(random_loi)
                # break
            else:
                # Get new LOI
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                curr_loi = lois.pop()
                logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))
                path = planner.plan(pose[:2], curr_loi)
                path.reverse() # reverse so closest wp is last so that pop() is cheap
                print(path)
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path:
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))

                # Calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
                ang_diff = -(ang_to_wp - pose[2]) # body frame

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360

                logging.getLogger('Navigation').debug('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < REACHED_THRESHOLD_M:
                    logging.getLogger('Navigation').info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                # Send command to robot
                logging.getLogger('Navigation').info(f"x {vel_cmd[0]}, z {vel_cmd[1]}")

                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])

            else:
                logging.getLogger('Navigation').info('End of path.')
                curr_loi = None
                for i in range(8):
                    pose, clues = loc_service.get_pose()
                    pose = pose_filter.update(pose)
                    print('pose', pose)
                    img = robot.camera.read_cv2_image(strategy='newest')

                    targets = cv_service.targets_from_image(img)

                    robot.chassis.drive_speed(x=0.0, y=0.0, z=30)  # set stop for safety
                    time.sleep(1.6)
                    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
                    time.sleep(1)

                    print('end of turn')
                    # if targets:  # rmb to put back
                    logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                    logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))

                    # For viz
                    for d in targets:
                        x, y, w, h = list(map(int, d.bbox))
                        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0), 2)
                        cv2.circle(img, (x, y), 1, (0, 255, 0))
                        cv2.putText(img, f'{CAT_2_NAME[d.cls]}', (x+int(w/2)+10,y+int(h/2)), 0, 0.3, (0,255,0))
                    cv2.imwrite(f"./data/imgs/det{i}.jpg", img)

                # TODO: Rotate all directions to capture target
                continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()
