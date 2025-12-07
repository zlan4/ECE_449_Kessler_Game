"""
Fuzzy Controller for Group 15:
==============================
Provides a fuzzy controller that can use learned values from a genetic algorithm
to optimize its fuzzy sets. This fuzzy controller takes an optional chromosome
argument and when it is not specified, then all fuzzy sets use predefined values,
otherwise it will take the fuzzy set parameters from this chromosome.
    If this file is executed as a script, then a genetic algorithm can be executed
with configurable population size and generation goal parameters, for example:

python group15_controller -p 10

will execute a genetic algorithm using a population size of 10 and

python group15_controller -g 30

will execute a genetic algorithm having a generation goal of 30. Population size
can be specified in a range from 1 to 50 and the generation goal can be specified
in a range from 1 to 10,000. If neither of these arguments are provided, then a
genetic algorithm with a population size and generation goal of 20 will be run.
    After the genetic algorithm has been executed, it will create a file called
solution.dat in the directory this script is placed in which the information about
the best chromosome found will be saved including its fitness, the population size
used, and the generation goal on its first line, then the genes on all following
lines. When the fuzzy controller class is used, it will search for this solution
file automatically and pull the gene data from it.
"""

import os
import math
import argparse
import random
import time
import traceback

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import EasyGA as GA

# Only imported for type hinting
from EasyGA.structure import Chromosome, Gene
from kesslergame.state_models import GameState, ShipState, AsteroidView

from kesslergame import KesslerController, Scenario, TrainerEnvironment


BULLET_TIME_UNIVERSE  = np.linspace(0.0, 1.7, 1001)
THETA_DELTA_UNIVERSE  = np.linspace(-math.pi, math.pi, 181)
THREAT_LEVEL_UNIVERSE = np.linspace(0.0, 1.0, 11)
SHIP_THRUST_UNIVERSE  = np.linspace(-480.0, 480.0, 100)
SHIP_TURN_UNIVERSE    = np.linspace(-180.0, 180.0, 361)

# The best chromosome found with the genetic algorithm.
# Found using a population size of 10 and a generation
# goal of 5.
BEST_FOUND_CHROMOSOME = [0.019528449375025857, 0.6836789292667184, -0.045383718578542454, -0.012559110824473002, 0.05755253405422077, 0.11240539084632029, 0.3910522164808833, -459.61306487449735, -343.8552128918893, -114.13965417420275, -121.4218357846888, -108.36596259401675, -75.17841825401896, -69.25971003283578, 31.752692017470622]


class FuzzyController(KesslerController):
    def __init__(self, chromosome: Chromosome = None):
        """Initializes a fuzzy controller with an optional chromosome.

        If the chromosome is specified, then it must provide at least 78 genes
        to populate all fuzzy sets. All genes after the 78th will be ignored.

        Arguments:
            chromosome: The chromosome containing fuzzy set parameters.
        """
        self.eval_frames = 0

        # Input variables
        # ===============
        # Bullet time is the amount of time the bullet needs to reach its
        # target and is also an approximation for the distance to the asteroid.
        bullet_time = ctrl.Antecedent(BULLET_TIME_UNIVERSE, 'bullet_time')
        # Theta delta is the amount the ship needs to turn to complete its
        # next action.
        theta_delta = ctrl.Antecedent(THETA_DELTA_UNIVERSE, 'theta_delta')
        # Threat level is a threat detection number that combines the distance,
        # size, and velocity of asteroids along with taking into account any
        # asteroid that will crash into the ship.
        threat_level = ctrl.Antecedent(THREAT_LEVEL_UNIVERSE, 'threat_level')

        # Output variables
        # ================
        ship_thrust = ctrl.Consequent(SHIP_THRUST_UNIVERSE,      'ship_thrust')
        ship_turn   = ctrl.Consequent(SHIP_TURN_UNIVERSE,        'ship_turn')
        ship_fire   = ctrl.Consequent(np.arange(-1.0, 1.0, 0.1), 'ship_fire')
        ship_mine   = ctrl.Consequent(np.arange(-1.0, 1.0, 0.1), 'ship_mine')

        # We want to prioritize the chromosome passed in as we could be
        # attempting to run the genetic algorithm while the solution.dat
        # file already exists.
        if os.path.isfile(SOLUTION_PATH) and chromosome is None:
            with open(SOLUTION_PATH, 'r') as file:
                lines = file.readlines()
                training_data = lines[0].split(",")
                print(f"Solution exists in solution.dat whose fitness is {training_data[0]} (fitness is between 0.0 and 1.0).")
                print(f"This was obtained using a population size of {training_data[1]} and a generation goal of {training_data[2].strip('\n')}.")
                chromosome = Chromosome([float(value) for value in lines[1:]])

        # Comment this line to use default values or to use a different set of values
        chromosome = Chromosome(BEST_FOUND_CHROMOSOME)
        # Fallback values for if the genetic algorithm has not been run.
        if chromosome is None:
            # Bullet time sets
            bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0.0, 0.0, 0.2])
            bullet_time['S']  = fuzz.trimf(bullet_time.universe, [0.0, 0.2, 0.5])
            bullet_time['M']  = fuzz.trimf(bullet_time.universe, [0.2, 0.5, 1.0])
            bullet_time['L']  = fuzz.smf(bullet_time.universe,    0.5, 1.0)

            # Theta delta sets, we use zmf and smf to account for all angles
            # larger than 6° because this is the largest angle we can move in
            # a single game update.
            theta_delta['NL'] = fuzz.zmf(theta_delta.universe,    -math.pi/30, -math.pi/45)
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, -math.pi/45, -math.pi/90])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/45, -math.pi/90,  math.pi/90])
            theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [-math.pi/90,  0,           math.pi/90])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/90,  math.pi/90,  math.pi/45])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [ math.pi/90,  math.pi/45,  math.pi/30])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,     math.pi/45,  math.pi/30)

            threat_level['L']  = fuzz.trimf(threat_level.universe, [0.0,  0.0, 0.25])
            threat_level['M']  = fuzz.trimf(threat_level.universe, [0.0,  0.3, 0.6])
            threat_level['H']  = fuzz.trimf(threat_level.universe, [0.4,  0.7, 1.0])
            threat_level['VH'] = fuzz.trimf(threat_level.universe, [0.75, 1.0, 1.0])

            # Output sets for thrust, due to the drag coefficient, any acceleration
            # below 80.0 m/s^2 will cause the ship to have zero acceleration, since
            # net acceleration is thrust + drag, where drag = 80 m/s^2. See ship.py
            ship_thrust['Reverse']     = fuzz.trimf(ship_thrust.universe, [-500.0, -500.0, -300.0])
            ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-400.0, -225.0,  -50.0])
            ship_thrust['Zero']        = fuzz.trimf(ship_thrust.universe, [ -80.0,    0.0,   80.0])
            ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [  50.0,  225.0,  400.0])
            ship_thrust['Forward']     = fuzz.trimf(ship_thrust.universe, [ 300.0,  500.0,  500.0])

            # Output sets for turn rate
            ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
            ship_turn['MedRight']  = fuzz.trimf(ship_turn.universe, [-180, -120,  -60])
            ship_turn['Right']     = fuzz.trimf(ship_turn.universe, [-120,  -60,   60])
            ship_turn['Zero']      = fuzz.trimf(ship_turn.universe, [ -60,    0,   60])
            ship_turn['Left']      = fuzz.trimf(ship_turn.universe, [ -60,   60,  120])
            ship_turn['MedLeft']   = fuzz.trimf(ship_turn.universe, [  60,  120,  180])
            ship_turn['HardLeft']  = fuzz.trimf(ship_turn.universe, [ 120,  180,  180])
        else:
            chromosome = [gene.value for gene in chromosome]
            bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0.0,           0.0,           chromosome[0]])
            bullet_time['S']  = fuzz.trimf(bullet_time.universe, [0.0,           chromosome[0], chromosome[1]])
            bullet_time['M']  = fuzz.trimf(bullet_time.universe, [chromosome[0], chromosome[1], 1.7])
            bullet_time['L']  = fuzz.smf(bullet_time.universe,    chromosome[1], 1.7)

            theta_delta['NL'] = fuzz.zmf(theta_delta.universe,    -math.pi/30,   -math.pi/45)
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi/30,   -math.pi/45,   chromosome[2]])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/45,   chromosome[2], chromosome[3]])
            theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [chromosome[2], chromosome[3], chromosome[4]])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [chromosome[3], chromosome[4], math.pi/45])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [chromosome[4], math.pi/45,    math.pi/30])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,    math.pi/45,    math.pi/30)

            threat_level['L']  = fuzz.trimf(threat_level.universe, [0.0,           0.0,           chromosome[5]])
            threat_level['M']  = fuzz.trimf(threat_level.universe, [0.0,           chromosome[5], chromosome[6]])
            threat_level['H']  = fuzz.trimf(threat_level.universe, [chromosome[5], chromosome[6], 1.0])
            threat_level['VH'] = fuzz.trimf(threat_level.universe, [chromosome[6], 1.0,           1.0])

            ship_thrust['Reverse']     = fuzz.trimf(ship_thrust.universe, [-500.0,        -500.0,        chromosome[7]])
            ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-500.0,        chromosome[7], chromosome[8]])
            ship_thrust['Zero']        = fuzz.trimf(ship_thrust.universe, [chromosome[7], chromosome[8], chromosome[9]])
            ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [chromosome[8], chromosome[9], 500.0])
            ship_thrust['Forward']     = fuzz.trimf(ship_thrust.universe, [chromosome[9], 500.0,         500.0])

            ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [-180,           -180,           chromosome[10]])
            ship_turn['MedRight']  = fuzz.trimf(ship_turn.universe, [-180,           chromosome[10], chromosome[11]])
            ship_turn['Right']     = fuzz.trimf(ship_turn.universe, [chromosome[10], chromosome[11], chromosome[12]])
            ship_turn['Zero']      = fuzz.trimf(ship_turn.universe, [chromosome[11], chromosome[12], chromosome[13]])
            ship_turn['Left']      = fuzz.trimf(ship_turn.universe, [chromosome[12], chromosome[13], chromosome[14]])
            ship_turn['MedLeft']   = fuzz.trimf(ship_turn.universe, [chromosome[13], chromosome[14], 180])
            ship_turn['HardLeft']  = fuzz.trimf(ship_turn.universe, [chromosome[14], 180,            180])

        # Output sets for fire and mine
        ship_fire['Yes'] = fuzz.trimf(ship_fire.universe, [ 0,  1, 1])
        ship_fire['No']  = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_mine['Yes'] = fuzz.trimf(ship_mine.universe, [ 0,  1, 1])
        ship_mine['No']  = fuzz.trimf(ship_mine.universe, [-1, -1, 0])

        # Fuzzy rules
        # ===========
        rules = []

        rules.append(ctrl.Rule((theta_delta['NM'] | theta_delta['NS']) & (threat_level['L'] | threat_level['M']), (ship_thrust['Zero'],        ship_turn['Right'],    ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule((theta_delta['PM'] | theta_delta['PS']) & (threat_level['L'] | threat_level['M']), (ship_thrust['Zero'],        ship_turn['Left'],     ship_fire['Yes'], ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NL'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PL'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NS'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['HardRight'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['Z'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PS'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['HardLeft'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NL'] & (threat_level['L'] | threat_level['M']),  (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PL'] & (threat_level['L'] | threat_level['M']),  (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & threat_level['VH'], (ship_mine['Yes'])))

        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NM'],                                             (ship_thrust['SlowReverse'], ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NS'],                                             (ship_thrust['SlowReverse'], ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['Z'],                                              (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PS'],                                             (ship_thrust['SlowReverse'], ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PM'],                                             (ship_thrust['SlowReverse'], ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NL'] & (threat_level['L'] | threat_level['M']),   (ship_thrust['Zero'],        ship_turn['HardLeft'],  ship_fire['Yes'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NL'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['Zero'],        ship_turn['HardRight'], ship_fire['Yes'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PL'] & (threat_level['L'] | threat_level['M']),   (ship_thrust['Forward'],     ship_turn['HardLeft'],  ship_fire['Yes'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PL'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['Yes'],  ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NL'],                                             (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['Yes'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NM'],                                             (ship_thrust['SlowForward'], ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NS'],                                             (ship_thrust['SlowForward'], ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['Z'],                                              (ship_thrust['SlowForward'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PS'],                                             (ship_thrust['SlowForward'], ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PM'],                                             (ship_thrust['SlowForward'], ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PL'],                                             (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['Yes'],  ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NL'],                                             (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NM'],                                             (ship_thrust['Forward'],     ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NS'],                                             (ship_thrust['Forward'],     ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['Z'],                                              (ship_thrust['Forward'],     ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PS'],                                             (ship_thrust['Forward'],     ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PM'],                                             (ship_thrust['Forward'],     ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PL'],                                             (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['Yes'], ship_mine['No'])))

        self.control_system = ctrl.ControlSystem(rules)


    def find_most_dangerous_asteroid(self, ship_state: ShipState, game_state: GameState) -> tuple[AsteroidView, float]:
        """Find the most dangerous asteroid based on distance, size, and velocity.

        Attempts to prioritize any asteroid that will intercept the ship,
        prioritizing larger and faster asteroids, but if none will, then it
        will prioritize based on distance.

        Arguments:
            ship_state: The state of the ship to find the most dangerous
                asteroid for.
            game_state: The current state of the game containing information
                about all asteroids.

        Returns:
            (most_dangerous_asteroid, threat_level): A tuple containing the
                predicted most dangerous asteroid for the given ship based
                on the current state of the game and the threat level of
                this asteroid.
        """
        most_dangerous = None
        highest_threat = -1

        for asteroid in game_state["asteroids"]:
            ship_position = Vec2D(ship_state["position"])
            asteroid_position = Vec2D(asteroid["position"])
            asteroid_velocity = Vec2D(asteroid["velocity"])
            dist_vec = ship_position - asteroid_position

            # Threat calculation: closer, larger, faster = more dangerous
            distance_threat = max(0, 1.0 - dist_vec.magnitude() / 1300.0)

            if asteroid_velocity.magnitude() > 0.0:
                angle_between = math.acos(min(dist_vec.dot_prod(asteroid_velocity) / (dist_vec.magnitude() * asteroid_velocity.magnitude()), 1.0))
                angle_between = angle_between * 180.0 / math.pi

                # Not guaranteed to hit the ship, but gives a good estimate on
                # which asteroids if a collision could happen
                if angle_between < 15.0:
                    intercept_time = dist_vec.magnitude() / asteroid_velocity.magnitude()
                    intercept_threat = max(0, 1 - intercept_time / 100.0)
                    size_threat = asteroid["size"] / 4.0
                    velocity_threat = min(1.0, asteroid_velocity.magnitude() / 200.0)

                    # Combined threat score where we prioritize asteroids that will
                    # intercept the ship.
                    threat_score = (0.5 * intercept_threat + 0.4 * distance_threat + 0.075 * size_threat + 0.025 * velocity_threat)
                else:
                    # When the asteroid won't hit the ship, it's best to only
                    # prioritize the distance, otherwise the ship may prioritize
                    # further away but larger asteroids that don't make sense
                    # to target at that point in time.
                    threat_score = distance_threat
            else:
                threat_score = distance_threat

            if threat_score > highest_threat:
                highest_threat = threat_score
                most_dangerous = asteroid

        return most_dangerous, highest_threat


    def calculate_intercept(self, ship_state: ShipState, asteroid: AsteroidView):
        """Finds the distance between the ship and given asteroid.

        Arguments:
            ship_state: The current state of the ship that will be used to
                determine the shooting angle.
            asteroid: The asteroid to be targeted by the given ship.

        Returns:
            (intercept_time, dist_between): A tuple containing the intercept
                time and distance between the ship and the given asteroid. If
                `use_ship` is `True`, then it is the amount of time it will to
                take for the asteroid to intersect with the ship, or 0.0 when
                it will not. If `use_ship` is `False`, then it is the amount of
                time it will take for a bullet to intersect the given asteroid.
        """
        ship_position     = Vec2D(ship_state["position"])
        asteroid_position = Vec2D(asteroid["position"])
        asteroid_velocity = Vec2D(asteroid["velocity"])

        # Determine the vector that points from the asteroid to the ship.
        asteroid_ship_vec = ship_position - asteroid_position
        asteroid_ship_theta = asteroid_ship_vec.direction()
        asteroid_direction = asteroid_velocity.direction()

        # Find the angle between the vector that points to the ship and
        # the asteroid's velocity vector. If we assume A = asteroid_ship_vec
        # and B = asteroid_velocity, then this is equivalent to:
        # intercept_angle = arccos(A * B / |A||B|)
        intercept_angle = asteroid_ship_theta - asteroid_direction

        # Apply the cosine to the value to cancel out the arccos operation
        cos_intercept = math.cos(intercept_angle)

        asteroid_vel = asteroid_velocity.magnitude()
        ship_ast_distance = asteroid_ship_vec.magnitude()
        obj_speed = 800.0

        # Applying the Law of Cosines on the triangle formed by the ship
        # velocity, asteroid velocity, and ship-asteroid distance vectors,
        # then transform into a quadratic equation at^2 + bt + c = 0.
        quadratic_coeff = asteroid_vel**2 - obj_speed**2
        linear_coeff = -2 * ship_ast_distance * asteroid_vel * cos_intercept
        const_term = ship_ast_distance**2
        targ_discriminant = linear_coeff**2 - 4*quadratic_coeff*const_term

        # No intersection occurs if the discriminant is negative.
        if targ_discriminant < 0 or quadratic_coeff == 0:
            return 0.0, ship_ast_distance

        sqrt_discriminant = math.sqrt(targ_discriminant)
        t1_intercept = (-linear_coeff + sqrt_discriminant) / (2 * quadratic_coeff)
        t2_intercept = (-linear_coeff - sqrt_discriminant) / (2 * quadratic_coeff)

        # Select the time intercept that is closer to zero, i.e., it happens
        # sooner in time than the other.
        if t1_intercept > t2_intercept:
            if t2_intercept >= 0:
                intercept_time = t2_intercept
            else:
                intercept_time = t1_intercept
        else:
            if t1_intercept >= 0:
                intercept_time = t1_intercept
            else:
                intercept_time = t2_intercept

        return intercept_time, ship_ast_distance


    def determine_turn_rate(self, intercept_time: float, ship_state: ShipState, asteroid: AsteroidView) -> float:
        """"Determines the angle the ship must rotate to be able to shoot the given asteroid.

        Arguments:
            intercept_time: The amount of time it will take a bullet to
                intercept the ship.
            ship_state: The current state of the ship including its position
                and velocity.
            asteroid: The asteroid the ship is attempting to shoot.

        Returns:
            shooting_theta: The amount the ship must rotate to shoot the given asteroid.
        """
        ship_curr_position = Vec2D(ship_state["position"])
        ship_velocity      = Vec2D(ship_state["velocity"])
        ship_position      = ship_curr_position + ship_velocity * (1/30)
        asteroid_position  = Vec2D(asteroid["position"])
        asteroid_velocity  = Vec2D(asteroid["velocity"])
        intercept = asteroid_position + (intercept_time + 1/30) * asteroid_velocity
        ship_intercept_angle = math.atan2((intercept.y - ship_position.y), (intercept.x - ship_position.x))

        # The amount we need to turn the ship to aim at where we want to shoot
        shooting_theta = ship_intercept_angle - (math.pi/180 * ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        return shooting_theta


    def actions(self, ship_state: ShipState, game_state: GameState) -> tuple[float, float, bool, bool]:
        """Main controller method called each time step when the game updates its states.

        Arguments:
            ship_state: The state of the ship being updated.
            game_state: The current state of the game.

        Returns:
            (`thrust`, `turn_rate`, `should_fire`, `should_drop_mine`): A tuple
                containing the new thrust rate and new turn rate of the ship
                along with two Boolean flags for if the ship should fire and if
                it should drop a mine.
        """
        # Find the most dangerous asteroid
        target_asteroid, threat_level = self.find_most_dangerous_asteroid(ship_state, game_state)

        if target_asteroid is None:
            return 100.0, 0.0, False, False

        # Calculate intercept parameters
        bullet_t, distance = self.calculate_intercept(ship_state, target_asteroid)
        shooting_theta = self.determine_turn_rate(bullet_t, ship_state, target_asteroid)

        # Create control system simulation
        controller = ctrl.ControlSystemSimulation(self.control_system, flush_after_run=1)

        controller.input['bullet_time'] = min(bullet_t, 0.99) if bullet_t else 1.0
        controller.input['theta_delta'] = shooting_theta
        controller.input['threat_level'] = threat_level

        try:
            controller.compute()
            thrust = float(controller.output['ship_thrust'])
            turn_rate = float(controller.output['ship_turn'])
            fire = bool(controller.output['ship_fire'] >= 0)
            drop_mine = bool(controller.output['ship_mine'] >= 0)

        except Exception as e:
            # Safe fallback behavior
            if distance < 200:  # Too close - evade
                thrust = float(-100.0)
                turn_rate = float(0.0)
                fire = bool(False)
                drop_mine = bool(True)
            elif bullet_t and bullet_t < 0.1:  # Good shot - take it
                thrust = float(0.0)
                turn_rate = float(0.0)
                fire = bool(True)
                drop_mine = bool(False)
            else:  # Default behavior
                thrust = float(100.0)
                turn_rate = float(0.0)
                fire = bool(False)
                drop_mine = bool(False)

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "Fuzzy Controller"


class Vec2D:
    def __init__(self, components: tuple[int | float, int | float] = None, x: int | float = None, y: int | float = None):
        """Creates a vector from either a tuple or a set of x and y values.

        Either the tuple must be specified or the x and y values. If both are
        given, then a `ValueError` will be raised.

        Arguments:
            components: A tuple containing an x-value and a y-value representing
                the x and y components of the vector.
            x: The vector's x component value.
            y: The vector's y component value.

        Raises:
            ValueError: If both `a_tuple` and `x` or `y` are specified.
        """
        if components and (x or y):
            raise ValueError("Cannot provide both a tuple and coordinates to form a vector!")

        if components:
            self._x = components[0]
            self._y = components[1]
        else:
            self._x = x
            self._y = y


    @property
    def x(self) -> int | float:
        return self._x


    @property
    def y(self) -> int | float:
        return self._y


    def magnitude(self) -> float:
        """Returns the magnitude of the vector."""
        return math.sqrt(self._x**2 + self._y**2)


    def direction(self) -> float:
        """Returns the angle between the components of the vector.

        Measures the angle in the range [-pi, pi] relative to the positive
        x-axis. If the angle is below the x-axis then the angle is negative,
        otherwise it is positive.
        """
        return math.atan2(self._y, self.x)


    def dot_prod(self, other: 'Vec2D') -> int | float:
        """Returns the dot product of the vector with the given vector."""
        return self._x * other._x + self._y * other._y


    def __mul__(self, other: int | float) -> 'Vec2D':
        """A new vector whose coordinates are scaled by the given `other` value."""
        return Vec2D(x=self._x * other, y=self._y * other)


    def __rmul__(self, other: int | float) -> 'Vec2D':
        """A new vector whose coordinates are scaled by the given `other` value."""
        return Vec2D(x=self._x * other, y=self._y * other)


    def __add__(self, other: 'Vec2D') -> 'Vec2D':
        """A new vector whose coordinates are sum of the vectors."""
        return Vec2D(x=self._x + other._x, y=self._y + other._y)


    def __sub__(self, other: 'Vec2D') -> 'Vec2D':
        """A new vector whose coordinates are the difference of the vectors.

        Produces a vector whose coordinates are the difference between this
        vector's coordinates and the provided other vector's coordinates. If
        this vector is vector A and other is vector B, then the resulting
        vector will be the vector pointing from B to A.
        """
        return Vec2D(x=self._x - other._x, y=self._y - other._y)


    def __str__(self) -> str:
        return f"<x={self._x:0.4f}, y={self._y:.4f}>"    


# =============================================================================
# Genetic Algorithm
# =============================================================================
genetic_start_time = 0
time_training = False


def main(population_size: int, generations: int):
    """Runs a genetic algorithm that seeks to optimize the fuzzy sets for the Kessler game.

    Arguments:
        population_size: The size of the genetic algorithm's population in each
            generation.
        generations: The number of generations to evolve the algorithm over.
    """
    global genetic_start_time
    asteroids_ga = GA.GA()
    asteroids_ga.fitness_goal = 'max'
    asteroids_ga.population_size = population_size
    asteroids_ga.generation_goal = generations
    asteroids_ga.chromosome_length = 15
    asteroids_ga.fitness_function_impl = ga_fitness
    asteroids_ga.chromosome_impl = ga_chromosome
    asteroids_ga.mutation_individual_impl = mutation

    if time_training:
        genetic_start_time = time.time()

    print("Begnning evolution...")
    asteroids_ga.evolve()
    best_chromosome = asteroids_ga.population[0]
    print("Evolution finished...\n")
    print("The best chromosome discovered was:")
    print(best_chromosome)

    if time_training:
        print(f"Total time elapsed: {format_time(time.time() - genetic_start_time, True)}")

    with open(SOLUTION_PATH, 'w') as file:
        file.write(str(best_chromosome.fitness) + "," + str(population_size) + "," + str(generations) + '\n')

        for idx, gene in enumerate(best_chromosome):
            if idx == (len(best_chromosome) - 1):
                file.write(str(gene.value))
            else:
                file.write(str(gene.value) + '\n')


def mutation(ga: GA.GA, chromosome: Chromosome):
    """Applies a random mutation to the given chromosome.

    If a mutation cannot be applied, then the given chromosome will be the
    chromosome returned.

    Arguments:
        ga: The genetic algorithm being used.
        chromosome: The chromosome to apply a mutation to.
    """
    index = random.randrange(len(chromosome))
    retry_mutation = True

    # Nudge the existing value of a gene based on the maximum value it can take
    # on in its universe. If the new value is outside the universe, then retry
    # the mutation.
    while retry_mutation:
        mutation_distance = 0.1
        gene_value = chromosome[index].value

        try:
            if 0 <= index <= 1:
                mutation_distance *= max(BULLET_TIME_UNIVERSE)
                gene_value += np.random.uniform(-mutation_distance, mutation_distance)

                if gene_value < min(BULLET_TIME_UNIVERSE) or gene_value > max(BULLET_TIME_UNIVERSE):
                    continue
            elif 2 <= index <= 4:
                # We use pi/45 here rather than the universe size because the ship cannot
                # rotate faster than pi/30 radians per tick and pi/45 is the next lowest
                # value in the fuzzy sets for a maximum value.
                mutation_distance *= (2 * math.pi/45)
                gene_value += np.random.uniform(-mutation_distance, mutation_distance)

                if gene_value < -math.pi/45 or gene_value > math.pi/45:
                    continue
            elif 5 <= index <= 6:
                mutation_distance *= max(THREAT_LEVEL_UNIVERSE)
                gene_value += np.random.uniform(-mutation_distance, mutation_distance)

                if gene_value < min(THREAT_LEVEL_UNIVERSE) or gene_value > max(THREAT_LEVEL_UNIVERSE):
                    continue
            elif 7 <= index <= 9:
                mutation_distance *= max(SHIP_THRUST_UNIVERSE)
                gene_value += np.random.uniform(-mutation_distance, mutation_distance)

                if gene_value < min(SHIP_THRUST_UNIVERSE) or gene_value > max(SHIP_THRUST_UNIVERSE):
                    continue
            elif 10 <= index <= 14:
                mutation_distance *= max(SHIP_TURN_UNIVERSE)
                gene_value += np.random.uniform(-mutation_distance, mutation_distance)

                if gene_value < min(SHIP_TURN_UNIVERSE) or gene_value > max(SHIP_TURN_UNIVERSE):
                    continue

            # Check to see if the gene we are applying a mutation to is not an minimum
            # value gene
            if (index not in (0, 2, 5, 7, 10)):
                if gene_value < chromosome[index - 1].value:
                    continue

            # Check to see if the gene we are applying a mutation to is not an maximum
            # value gene
            if (index not in (1, 4, 6, 9, 14)):
                if gene_value > chromosome[index + 1].value:
                    continue

            retry_mutation = False
        except Exception as e:
            print(f"Attempting to mutate gene {index} caused an issue:", e)
            print(traceback.format_exc())
            return chromosome
 
    chromosome[index] = Gene(gene_value)
    return chromosome


def ga_fitness(chromosome: Chromosome) -> float:
    """Fitness function for the Kessler game.

    Fitness is determined by running a single game and measuring the
    ratio of the number of asteroids destroyed out of the total possible
    number of asteroids in the game (400), the accuracy ratio, and the ratio
    of the number of deaths out of the possible amount of deaths (3). These
    ratios are then combined in a weighted sum.
    """
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=10,
                                ship_states=[
                                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)
    fitness = -1.0

    try:
        controller = FuzzyController(chromosome)
        game = TrainerEnvironment()
        total_asteroids = 0
        average_accuracy = 0.0
        total_deaths = 0
        total_bullets_shot = 0
        total_bullets_hit = 0
        num_games = 2

        for _ in range(num_games):
            score, _ = game.run(scenario=my_test_scenario, controllers=[controller])
            team = score.teams[0]
            total_asteroids += team.asteroids_hit
            total_bullets_shot += team.shots_fired
            total_bullets_hit += team.bullets_hit
            total_deaths += team.deaths

        # We start with 10 asteroids with a size of 4 in each game. Each can
        # break into 3 smaller asteroids until they reach size 1 giving us
        # 10 * SUM(3^n) for n=0 to n = 3 asteroids which is 400 in total in
        # one game.
        fraction_asteroids = total_asteroids / (400.0 * num_games)
        fraction_deaths = 1 - (total_deaths / (3.0 * num_games))
        average_accuracy = total_bullets_hit / total_bullets_shot

        # These three terms are the most important to the game's score with the
        # number of asteroids being the most important and accuracy close by (for
        # breaking ties). Deaths are less important but still important for making
        # the game last long.
        fitness = 0.5 * fraction_asteroids + 0.35 * average_accuracy + 0.15 * fraction_deaths
        print(f"Fitness: {fitness}, Asteroids: {total_asteroids}, Accuracy: {average_accuracy}, Deaths: {total_deaths}")
        print("Chromosome Data:")
        print(chromosome)
    except Exception as e:
        print("Caught error when finding fitness:", e)
        print("Chromosome that caused exception:")
        print(chromosome)
        print(traceback.format_exc())
        # If for some reason we generate an invalid fuzzy set, then give a
        # penalty to the chromosome.
        fitness = -1.0

    if time_training:
        print(f"Total elapsed time: {format_time(time.time() - genetic_start_time, True)}\n")

    return fitness

# BM1
def ga_chromosome() -> list[float]:
    """Generates a new chromosome for the Kessler game fuzzy controller."""
    chromosome_data = []
    chromosome_data.extend(generate_points(BULLET_TIME_UNIVERSE, 2))
    chromosome_data.extend(generate_points([-math.pi/45, math.pi/45], 3))
    chromosome_data.extend(generate_points(THREAT_LEVEL_UNIVERSE, 2))
    chromosome_data.extend(generate_points(SHIP_THRUST_UNIVERSE, 3))
    chromosome_data.extend(generate_points(SHIP_TURN_UNIVERSE, 5))
    return chromosome_data


def generate_points(universe: np.ndarray | list, num_points: int):
    """Generates a given number of random points.

    The points generated do not necessarily correspond to the points in the
    universal set, but instead will fall between the maximum and minimum
    values.

    Arguments:
        universe: The set of values in the universe of discourse.
        num_points: The number of random points to generate.

    Returns:
        points: A list of randomly generated points that fall between the
            maximum and minimum values in `universe` sorted from lowest
            to highest.
    """
    min_point = float(min(universe))
    max_point = float(max(universe))
    points = []

    for _ in range(num_points):
        point = np.random.uniform(min_point, max_point)

        while point in points:
            point = np.random.uniform(min_point, max_point)

        points.append(point)

    points.sort()
    return points


def format_time(time_in_seconds: float, include_ms: bool=False):
    """Formats the given time in the format {H}h {M}m {S}s.

    If the given time is less than 1 second or `include_ms` is `True`, then
    the millisecond time will be appended to the time. By default, milliseconds
    are only included if the time is less than 1 second.

    Arguments:
        time_in_seconds: The amount of time in seconds.
        include_ms: If milliseconds should be included in the formatted time.
    """
    seconds = math.floor(time_in_seconds % 60)
    milliseconds = round((time_in_seconds - math.floor(time_in_seconds)) * 1000)
    time_string = ""

    if time_in_seconds < 60:
        time_string = f"{seconds}s"
    elif time_in_seconds < 3600:
        time_string = f"{int(time_in_seconds // 60)}m {seconds}s"
    else:
        time_string = f"{int(time_in_seconds // 3600)}h {int(time_in_seconds // 60)}m {seconds}s"

    if time_in_seconds < 1 or include_ms:
        time_string += f" {milliseconds}ms"

    return time_string



CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOLUTION_PATH = os.path.join(CURRENT_DIRECTORY, 'solution.dat')


if __name__ == "__main__":
    desc = ("A program that runs a genetic algorithm to find the best parameters for the Kessler Game."
            " Once the algorithm has been executed, a scenario can be executed using the fuzzy controller"
            " and it will obtain the parameters from the solution file.")
    parser = argparse.ArgumentParser(prog="group15_controller.py", description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--population", help="the number of individuals in the GA's population between 1 and 50", metavar="[1-50]", default=20, type=int)
    parser.add_argument("-g", "--generations", help="the number of generations to evolve the GA over between 1 and 10,000", metavar="[1-10000]", default=10, type=int)
    parser.add_argument("-t", "--timeit", help="if present, then timing information will be output as the genetic algorithm executes such as time passed and estimated time remaining", action="store_true")
    ns = parser.parse_args()

    if (ns.population < 1 or ns.population > 50) or (ns.generations < 1 or ns.generations > 10000):
        parser.print_usage()
    else:
        time_training = ns.timeit
        main(ns.population, ns.generations)