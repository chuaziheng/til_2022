import logging
import time
from typing import List

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things
from tilsdk.mock_robomaster.robot import Robot                 # Use this for the simulator
# from robomaster.robot import Robot                              # Use this for real robot

# Import your code
from cv_service import CVService, MockCVService
from nlp_service import NLPService
from planner import Planner
import cv2
import numpy as np

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
CAT_2_NAME = {1: 'Fallen', 2: 'Standing'}

def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    # cv_service = MockCVService(model_dir=CV_MODEL_DIR)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)
    loc_service = LocalizationService(host='localhost', port=5566)
    rep_service = ReportingService(host='localhost', port=5566)
    robot = Robot()
    robot.initialize(conn_type="sta")
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

    # Initialize tracker
    # TODO: Participant to tune PID controller values. Currently all 0 so it wont move when no goal
    tracker = PIDController(Kp=(0.0, 0.0), Kd=(0.0, 0.0), Ki=(0.0, 0.0))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5)

    # Define filter to exclude clues seen before
    new_clues = lambda c: c.clue_id not in seen_clues

    # Main loop
    while True:
        # Get new data
        # robot.chassis.drive_speed(x=0.5, y=0, z=0)
        pose, clues = loc_service.get_pose()
        pose = pose_filter.update(pose)
        img = robot.camera.read_cv2_image(strategy='newest')

        if not pose:
            # not new data, continue to next iteration
            continue
        clues = filter(new_clues, clues)

        # process clues using NLP and determine any new LOI
        if clues:
            new_lois = nlp_service.locations_from_clues(clues)
            # update_locations(lois, new_lois)  # TODO: implement update_locations
            lois += new_lois
            seen_clues.update([c.clue_id for c in clues])


        # process image and detect targets
        targets = cv_service.targets_from_image(img)

        # submit targets
        if targets:
            for d in targets:
                x, y, w, h = list(map(int, d.bbox))
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, f'{CAT_2_NAME[d.cls]}', (x+w+10,y+h), 0, 0.3, (0,255,0))
            cv2.imwrite("./data/imgs/det.jpg", img)
            logging.getLogger('Main').info(f"{len(targets)} targets detected.")
            logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))

        if not curr_loi:
            # if current point is not LOI
            if len(lois) == 0:
                logging.getLogger('Main').info('No more LOI')
                # TODO: perform random search for new clues or targets
            else:
                print('loi length ', len(lois))
                # Get new LOI
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                curr_loi = lois.pop()
                logging.getLogger('Main').info(f"Current LOI set to: {curr_loi}")

                # Plan a path to new LOI
                cur_location = RealLocation(pose.x, pose.y)
                logging.getLogger('Main').info(f'Planning path from {cur_location} to {curr_loi}')
                path = planner.plan(cur_location, curr_loi)  # TODO: debug
                path.reverse()
                print(path)
                curr_wp = None
                logging.getLogger('Main').info('Path planned')
        else:
            # There is a current LOI goal
            if path:
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info(f'New waypoint: {curr_wp}')

                # calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
                ang_diff = -(ang_to_wp - pose[2])  # body frame

                # ensure ang_diff is in [180, 180]
                if ang_diff < -180:
                    ang_diff += 360
                if ang_diff > 180:
                    ang_diff -= 360

                logging.getLogger('Navigation').debug(f'ang_to_wp: {ang_to_wp}, hdg: {pose[2]}, ang_diff: {ang_diff}')

                # consider waypoint reached if within a threshold distance
                if dist_to_wp < REACHED_THRESHOLD_M:
                    logging.getLogger('Navigation').info(f'Reached wp: {curr_wp}')
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # if robot is facing the wrong dir, turn to face waypoint first before moving forward
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                # send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])

            else:
                logging.getLogger('Navigation').info('End of path')
                curr_loi = None

                # TODO: perform search behavior
                continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()
