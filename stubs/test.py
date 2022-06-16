import logging
from numbers import Real
import time
from typing import List

from torch import true_divide

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things
# from tilsdk.mock_robomaster.robot import Robot                 # Use this for the simulator
from robomaster.robot import Robot                              # Use this for real robot

# Import your code
from cv_service import CVService, MockCVService
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
CAT_2_NAME = {1: 'Fallen', 2: 'Standing'}

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
        width, height = random.randint(map_.width), random.randint(map_.height)
        random_grid_loc = GridLocation(width, height)
        if map_.passable(random_grid_loc) and map_.in_bounds(random_grid_loc):
            break
    return map_.grid_to_real(random_grid_loc)

def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    # cv_service = MockCVService(model_dir=CV_MODEL_DIR)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)
    # loc_service = LocalizationService(host='localhost', port=5566)
    loc_service = LocalizationService(host='192.168.20.56', port=5522)
    rep_service = ReportingService(host='192.168.20.56', port=5522)

    robot = Robot()
    # robot.initialize(conn_type="sta")
    robot.initialize(conn_type="sta", sn="3JKDH2T001U0H4")  # for real robot
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
    tracker = PIDController(Kp=(0.25, 0.25), Kd=(0.0, 0.0), Ki=(0.0, 0.0))
    # tracker = PIDController(Kp=(0.0, 0.0), Kd=(0.0, 0.0), Ki=(0.0, 0.0))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5)

    # Define filter to exclude clues seen before
    new_clues = lambda c: c.clue_id not in seen_clues

   # Main loop
    # while True:
    for i in range(8):
        # Get new data
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
        # if targets:
        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))

        # For viz
        for d in targets:
            x, y, w, h = list(map(int, d.bbox))
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f'{CAT_2_NAME[d.cls]}', (x+w+10,y+h), 0, 0.3, (0,255,0))
        cv2.imwrite(f"./data/imgs/det{i}.jpg", img)

       



if __name__ == '__main__':
    main()
