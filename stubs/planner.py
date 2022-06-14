from tkinter import Grid
from typing import List, Tuple, TypeVar, Dict
from tilsdk.localization import *
import heapq
from queue import PriorityQueue

T = TypeVar('T')

class NoPathFoundException(Exception):
    pass

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class Planner:
    def __init__(self, map_:SignedDistanceGrid = None, sdf_weight:float = 0.0):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        '''
        self.map = map_
        self.sdf_weight = sdf_weight

    def update_map(self, map:SignedDistanceGrid):
        '''Update planner with new map.'''
        self.map = map

    def heuristic(self, a:GridLocation, b:GridLocation) -> float:
        '''Planning heuristic function

        Params
        ------
        a: GridLocation
            Starting Location
        b: GridLocation
            Goal location
        '''
        return euclidean_distance(a, b)


    def plan(self, start:RealLocation, goal:RealLocation) -> List[RealLocation]:
        '''Plan in real coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.

        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''

        path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal))
        return [self.map.grid_to_real(wp) for wp in path]

    def plan_grid(self, start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Plan in grid coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.

        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''

        if not self.map:
            raise RuntimeError('Planner map is not initialized.')

        # TODO: Participant to complete.
        frontier = PriorityQueue()
        frontier.put(start, 0)
        prev: Dict[GridLocation, GridLocation] = {}
        current_cost: Dict[GridLocation, float] = {}
        prev[start] = None
        current_cost[start] = 0

        while not frontier.is_empty():
            # TODO: Participant to complete
            pass

        if goal not in prev:
            raise NoPathFoundException
        return self.reconstruct_path(prev, start, goal)

    def reconstruct_path(self,
                        prev:Dict[GridLocation, GridLocation],
                        start:GridLocation,
                        goal:GridLocation) -> List[GridLocation]:
        '''Traces traversed locations to reconstruct path

        '''
        # TODO
        pass
