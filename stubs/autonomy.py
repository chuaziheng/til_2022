import logging
import time
from typing import List

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage # import optional useful things
from tilsdk.mock_robomaster.robot import Robot                 # Use this for the simulator
# from robomaster.robot import Robot                              # Use this for real robot

# Import your code
from cv_service import CVService, MockCVService
from nlp_service import NLPService
from planner import Planner
import cv2

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

    # TODO: process map?
    planner = Planner(map_, sdf_weight=0.5)
    print('planner', planner)

    # Initialize variables
    # TODO: If needed.

    # Initialize tracker
    # TODO: Participant to tune PID controller values.
    tracker = PIDController(Kp=(0.0, 0.0), Kd=(0.0, 0.0), Ki=(0.0, 0.0))

    ep_chassis = robot.chassis
    # Main loop
    while True:
        # Get new data
        ep_chassis.drive_speed(x=0.5, y=0, z=0)
        pose, clues = loc_service.get_pose()
        print('clues', len(clues))

        # NLP Service
        locs = nlp_service.locations_from_clues(clues)
        print('locs', locs)

        # TODO: Call planner to get to goal location

        # CV Service
        # TODO: robot to rotate, take pic and do inference

        img = robot.camera.read_cv2_image(strategy='newest')
        det = cv_service.targets_from_image(img)
        print('det', det)
        if det:
            for d in det:
                x, y, w, h = list(map(int, d.bbox))
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, f'{CAT_2_NAME[d.cls]}', (x+w+10,y+h), 0, 0.3, (0,255,0))
            cv2.imwrite("./data/imgs/det.jpg", img)
        # postprocess det

        time.sleep(1)

        # Submit detection

        # TODO: use tracker to continue exploring the grid



    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()
