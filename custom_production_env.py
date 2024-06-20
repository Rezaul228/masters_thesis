# Custome production environment for job-shop scheduling problem
import json
from enum import Enum
import random
import collections
import numpy as np
import csv


class SimParams:
    _instance = None
    N_ORDER = 0

    MAX_NUM_OF_AGENT_GEN = 3

    AGN_GEN_INTRVL_MIN = 0
    AGN_GEN_INTRVL_MAX = 2

    AGENT_ID = 0
    TOTAL_AGENT_COMPLETED = 0

    SIM_TIME = 1

    ALL_FINISHED_AGENTS = []

    @classmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls.ALL_FINISHED_AGENTS.clear()
        cls.N_ORDER = 0
        cls.AGENT_ID = 0
        cls.TOTAL_AGENT_COMPLETED = 0
        cls.SIM_TIME = 1


class GenerateSimData:
    SIM_DATA = {}

    m_info_key = "machineInfo"
    sim_data_key = "simData"
    agent_info_key = "agentInfo"
    stat_key = "statistics"

    OUTPUT_FILE = "sim_data.json"

    @staticmethod
    def generate_agent_info(all_agents):
        if GenerateSimData.agent_info_key not in GenerateSimData.SIM_DATA:
            GenerateSimData.SIM_DATA[GenerateSimData.agent_info_key] = {}

        for time, agents in all_agents.items():
            for agent in agents:
                if agent.agent_id not in GenerateSimData.SIM_DATA[GenerateSimData.agent_info_key]:
                    GenerateSimData.SIM_DATA[GenerateSimData.agent_info_key][agent.agent_id] = {}
                GenerateSimData.SIM_DATA[GenerateSimData.agent_info_key][agent.agent_id] = \
                    {
                        "hasPriority": agent.is_prioritized,
                        "priority": agent.priority.value,
                        "task_list": [task.machine_type.value for task in agent.task_list]
                    }

    def reset(self):
        GenerateSimData.SIM_DATA = {}

    @staticmethod
    def generate_task_status(agent_list):
        def get_total_waiting_time(tasks):
            count = 0

            for task in tasks:
                count += task.waiting_time

            return count

        """
            {
                "statistics": {
                    "<agent-id>": {
                        "entryTime": "",
                        "exitTime": "",
                        "duration": 0,
                        "taskStat": [
                            {
                                "machine": "",
                                "arrivalTime": "",
                                "serviceTime": "",
                                "finishTime": "",
                                "waitTime": ""
                            }
                        ]
                    }
                }
            }
        """
        if GenerateSimData.stat_key not in GenerateSimData.SIM_DATA:
            GenerateSimData.SIM_DATA[GenerateSimData.stat_key] = {}

        for agent in agent_list:
            if agent.agent_id not in GenerateSimData.SIM_DATA[GenerateSimData.stat_key]:
                GenerateSimData.SIM_DATA[GenerateSimData.stat_key][agent.agent_id] = {}

                GenerateSimData.SIM_DATA[GenerateSimData.stat_key][agent.agent_id] = {
                    "hasPriority": agent.is_prioritized,
                    "priorityLevel": -1 if not agent.is_prioritized else agent.priority.value,
                    "entryTime": agent.generation_time,
                    "exitTime": agent.finish_time,
                    "duration": agent.finish_time - agent.generation_time,
                    "totalWaitingTime": get_total_waiting_time(agent.task_list),
                    "taskStat": [
                        {
                            "machine": task.machine.name,
                            "arrivalTime": task.arrival_time,
                            "serviceTime": task.service_time,
                            "finishTime": task.finish_time,
                            "waitTime": task.waiting_time
                        } for task in agent.task_list
                    ]

                }

    @staticmethod
    def generate_machine_info():
        if GenerateSimData.m_info_key not in GenerateSimData.SIM_DATA:
            GenerateSimData.SIM_DATA[GenerateSimData.m_info_key] = {}

        for m_type, machines in env.machine_collection.items():
            for machine in machines:
                GenerateSimData.SIM_DATA[GenerateSimData.m_info_key][machine.name] = {"capacity": 1}

    @staticmethod
    def generate_time_specific_data(time, machine_pool):
        def get_machine_detail(agent_list):
            result = []
            for agent in agent_list:
                result.append({agent.agent_id: {
                    "hasPriority": agent.is_prioritized,
                    "allTaskFinished": agent.has_all_task_finished()
                }})
            return result

        if GenerateSimData.sim_data_key not in GenerateSimData.SIM_DATA:
            GenerateSimData.SIM_DATA[GenerateSimData.sim_data_key] = {}

        if time not in GenerateSimData.SIM_DATA[GenerateSimData.sim_data_key]:
            GenerateSimData.SIM_DATA[GenerateSimData.sim_data_key][time] = []

        state_vector = []
        for m_type, machines in machine_pool.items():

            for machine in machines:
                machine_info_dict = {machine.name: {}}
                machine_info_dict[machine.name]["QUEUE"] = get_machine_detail(machine.agents_in_queue)
                machine_info_dict[machine.name]["SERVING"] = get_machine_detail(machine.agents_in_service)
                machine_info_dict[machine.name]["FINISHED"] = get_machine_detail(machine.agents_finished)

                GenerateSimData.SIM_DATA[GenerateSimData.sim_data_key][time].append(machine_info_dict)
                machine.print_machine_info(with_finish=True)
                state_vector.append(machine.get_machine_state(time))
        #print("\tState vector:\n\t\t", state_vector)

    @staticmethod
    def write_to_file(path):
        with open(path, "w") as fp:
            import json
            fp.write(json.dumps(GenerateSimData.SIM_DATA, indent=4))
        fp.close()


class TaskStatus(Enum):
    GENERATED = "Generated"
    QUEUE = "InQueue"
    PROCESSING = "TaskIsProcess"
    FINISH = "TaskFinishedFromMachine"
    EXIT = "ExitFromResource"
    NOTREADY = "NotReadyForGenerate"


class Priority(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Task(object):
    def __init__(self, machine_type, machine=None, status=TaskStatus.GENERATED, all_machines=None):
        self.machine = machine
        self.status = status
        self.machine_type = machine_type
        # this will store all the machines of that specific given machine type
        self.machine_lists_of_this_type = Machine.get_these_machines(machine_type, all_machines)

        self.arrival_time = -1
        self.service_time = -1
        self.finish_time = -1
        self.waiting_time = -1

    def reset(self):
        self.status = TaskStatus.GENERATED
        self.machine = None
        self.machine_type = None

        self.arrival_time = -1
        self.service_time = -1
        self.finish_time = -1
        self.waiting_time = -1

    def update_timing_info(self, arrival_time=-1, serving_time=-1, finishing_time=-1, waiting_time=-1):
        if arrival_time != -1:
            self.arrival_time = arrival_time
        if serving_time != -1:
            self.service_time = serving_time
        if finishing_time != -1:
            self.finish_time = finishing_time
        if waiting_time != -1:
            self.waiting_time = waiting_time


class Agent:
    def __init__(self, agent_id, arrival_time, machines):
        self.agent_id = agent_id
        self.machines = machines

        self.generation_time = arrival_time
        self.arrival_time = 0
        self.serving_time = 0
        self.finish_time = 0
        self.waiting_time = 0

        self.priority = Priority.NONE

        self.is_prioritized = False #random.sample([True, False], 1)[0]

        if self.is_prioritized:
            priority_list = [priority for priority in Priority]
            priority_list.remove(Priority.NONE)
            self.priority = random.sample(priority_list, 1)[0]

        self.task_list = []

        # Generate random task for the agent
        self.__create_task_list_for_agent()

    def reset(self):
        for task in self.task_list:
            task.reset()

        self.arrival_time = 0
        self.serving_time = 0
        self.finish_time = 0
        self.waiting_time = 0

    def print_agent_vector(self):
        av = self.get_agent_vector()

        print(f"\n\tAgent vector of {self.agent_id}:")
        print(f"\t\tMachine types (0): {av[0]}")

        print(f"\t\tMachine in use (1.1): {av[1][0]}")
        print(f"\t\tMachine remaining time (1.2): {av[1][1]}")

        print(f"\t\tFilled queue of each machine (2): {av[2]}")

        print(f"\t\tPriority order indicator (3.1): {av[3][0]}")
        print(f"\t\tPriority order indicator (3.2): {av[3][1]}")

        print(f"\t\tPosition (x=machine_id,y=queue slot) (4){[]}")
        print(f"\t\t(5){[]}")

    def get_agent_vector(self):
        def get_list_of_machine_types():
            # 0
            return [index for index in range(len(MachineType))]

        def get_list_of_machine_usage(task_list):
            # 1.1
            machine_usage = []
            for task in task_list:
                occupation_list = []
                machines_of_this_type = env.get_machine_by_type(task.machine_type)
                for machine in machines_of_this_type:
                    status = 1 if machine.is_anyone_in_service() else 0
                    occupation_list.append(status)
                machine_usage.append(occupation_list)

            return machine_usage

        def get_remaining_time_of_all_machines_assigned(task_list):
            # 1.2
            remaining_time_list = []

            for task in task_list:
                machines_of_this_type = env.get_machine_by_type(task.machine_type)
                machines_of_this_task = []
                for machine in machines_of_this_type:
                    rem_time = machine.get_est_to_finish_excluding_agents_time()
                    machines_of_this_task.append(rem_time)
                remaining_time_list.append(machines_of_this_task)

            return remaining_time_list

        def get_queue_position(task_list):
            # 2
            queue_position_list = []

            num_of_agents = len(env.all_agents)

            for task in task_list:
                machines_of_this_type = env.get_machine_by_type(task.machine_type)
                task_wise_machine = []
                current_queue = [0] * num_of_agents
                for machine in machines_of_this_type:
                    for idx in range(len(machine.agents_in_queue)):
                        current_queue[idx] = 1
                    task_wise_machine.append(current_queue)
                queue_position_list.append(task_wise_machine)

            return queue_position_list

        def get_priority_list(task_list):
            # 3.1
            priority_info_list = []
            num_of_agents = len(env.all_agents)

            for task in task_list:
                machines_of_this_type = env.get_machine_by_type(task.machine_type)
                task_wise_machine = []
                current_queue = [0] * num_of_agents
                for machine in machines_of_this_type:
                    for idx in range(len(machine.agents_in_queue)):
                        agent = machine.agents_in_queue[idx]
                        current_queue[idx] = (1 if agent.is_prioritized else 0)
                    task_wise_machine.append(current_queue)
                priority_info_list.append(task_wise_machine)

            return priority_info_list

        """
        [
            0 - [list of machine types],
            1 - [
                    1.1 - [if a machine is in use or not],
                    1.2 - [remaining time to be free including queue]
                ],
            2 - [   filled queue position, if a queue is in use. fill queue with 0 int he beginning and the size should
                    be the number of total agents],
            3 - [
                3.1 - [ for each machine [if an agent in the queue has priority or not, 1/0]]
                ],
            4- (machine_id, either assigned or not),
            5 - []
        ]

        """
        return [
            get_list_of_machine_types(),
            [
                get_list_of_machine_usage(self.task_list),
                get_remaining_time_of_all_machines_assigned(self.task_list)
            ],
            get_queue_position(self.task_list),
            [get_priority_list(self.task_list),
             [],
             ],
            [],
            []
        ]

    def id_with_prior_info(self):
        if self.is_prioritized:
            pstr = "L"
            if self.priority == Priority.HIGH:
                pstr = "H"
            elif self.priority == Priority.MEDIUM:
                pstr = "M"
            return str(self.agent_id) + "(" + pstr + ")"
        return str(self.agent_id)

    def __create_task_list_for_agent(self):
        machine_types = Machine.get_machine_types()
        num_of_tasks = 2  # random.randint(1, len(machine_types))
        selected_machine_types = random.sample(machine_types, num_of_tasks)

        # for machine_type in selected_machine_types:
        # task = Task(machine_type, None, TaskStatus.GENERATED, self.machines)
        # self.task_list.append(task)

        # currently the task size is fixed by two.
        # we intentionally create two tasks with the same order
        self.task_list.append(Task(MachineType.SANDING, None, TaskStatus.GENERATED, self.machines))
        self.task_list.append(Task(MachineType.DRILL, None, TaskStatus.NOTREADY, self.machines))

    def get_task_by_machine_id(self, machine_id):
        for task in self.task_list:
            if task.status == TaskStatus.GENERATED:
                for machine in task.machine_lists_of_this_type:
                    if machine.machine_id == machine_id:
                        return task
        return None

    # Action to take
    def action(self, current_time, input_machine):

        state_vector = []
        for task in self.task_list:
            if task.status == TaskStatus.GENERATED:
                for machine in task.machine_lists_of_this_type:
                    state_vector.append(machine.get_machine_state(current_time))
        if not state_vector:
            return None

        desired_task: Task = self.get_task_by_machine_id(input_machine)
        if desired_task:
            desired_task.machine = Machine.get_machine_by_id(input_machine, self.machines)
            desired_task.status = TaskStatus.QUEUE
        return desired_task

    def get_next_task(self, current_time, show_printouts):
        def find_machine_state_with_min_es_time(vector):
            desired_state = None
            for state in vector:
                if desired_state is None:
                    desired_state = state
                    continue
                if state[6] < desired_state[6]:
                    desired_state = state
            return desired_state

        state_vector = []
        for task in self.task_list:
            if task.status == TaskStatus.GENERATED:
                for machine in task.machine_lists_of_this_type:
                    state_vector.append(machine.get_machine_state(current_time))
        if not state_vector:
            return None

        desired_machine_state: list = find_machine_state_with_min_es_time(state_vector)
        desired_task: Task = self.get_task_by_machine_id(desired_machine_state[0])
        if show_printouts:
            print(f"Desired Machine State: {desired_machine_state[0]}")
        if desired_task:
            desired_task.machine = Machine.get_machine_by_id(desired_machine_state[0], self.machines)
            desired_task.status = TaskStatus.QUEUE
        return desired_task

    def get_task(self, status):

        for task in self.task_list:
            if task.status == status:
                return task
        return None




    def has_all_task_finished(self):
        for task in self.task_list:
            if task.status != TaskStatus.FINISH:
                return False
        return True

    def has_served_by_any_machine(self):
        for task in self.task_list:
            if task.status != TaskStatus.GENERATED:
                return True
        return False

    def status_message(self):
        msg = f"\n\tAgent: {self.id_with_prior_info()}, "
        msg += f"{len(self.task_list)} task(s) ["

        index = 1
        task: Task
        for task in self.task_list:
            msg += f"\n\t\t{index}. Machine type: {task.machine_type.value}, " \
                   f"Machine: {task.machine.name if task.machine is not None else 'Not selected yet'} " \
                   f"({'AT: ' + str(task.arrival_time) if task.arrival_time != -1 else ''}" \
                   f"{', ST: ' + str(task.service_time) if task.service_time != -1 else ''}" \
                   f"{', WT: ' + str(task.waiting_time) if task.waiting_time != -1 else ''}" \
                   f"{', FT:' + str(task.finish_time) if task.finish_time != -1 else ''})" \
                   f" - {task.status.value}, "
            index += 1

        msg = msg[:-2]
        msg += f"]"

        return msg


class MachineType(Enum):
    SANDING = "Sand"
    DRILL = "Drill"

    # PAINT = "Paint"

    # CURVE = "CurvingMachine"

    @staticmethod
    def get_type_index(mtype):
        idx = 0
        for tp in MachineType:
            if tp == mtype:
                return idx
            idx += 1


class Machine:
    def __init__(self, machine_id=1, name="default", machine_type: MachineType = None, job_duration=0):
        self.machine_id = machine_id
        self.name = name
        self.machine_type: MachineType = machine_type
        self.job_duration = job_duration

        self.agents_in_queue = []
        self.agents_in_service = []
        self.agents_finished = []

    def reset(self):
        for agent in self.agents_in_queue:
            agent.reset()
        self.agents_in_queue.clear()

        for agent in self.agents_in_service:
            agent.reset()
        self.agents_in_service.clear()

        for agent in self.agents_finished:
            agent.reset()
        self.agents_finished.clear()

    @staticmethod
    def get_machine_types():
        type_of_machines = []
        for machine_type in MachineType:
            type_of_machines.append(machine_type)
        return type_of_machines

    @staticmethod
    def get_machine_by_id(machine_id, all_machines):
        for _, machines in all_machines.items():
            for machine in machines:
                if machine.machine_id == machine_id:
                    return machine

    @staticmethod
    def get_these_machines(machine_type: MachineType, all_machines):
        return all_machines[machine_type]

    def is_anyone_in_service(self):
        return True if len(self.agents_in_service) > 0 else False

    def is_anyone_prior_in_service(self):
        if len(self.agents_in_service) > 0:
            return self.agents_in_service[0].is_prioritized
        return False

    def get_queue_size(self):
        return len(self.agents_in_queue)

    def get_remaining_time_for_serving_agent(self, current_time):
        remaining_time = 0
        if self.is_anyone_in_service():
            agent_in_service = self.agents_in_service[0]
            remaining_time = agent_in_service.get_task(TaskStatus.PROCESSING).finish_time - current_time
        return remaining_time

    def get_priority_values_in_the_queue(self):
        priority_value_list = []
        if self.is_anyone_in_queue():
            for agent in self.agents_in_queue:
                if agent.is_prioritized:
                    priority_value_list.append(agent.priority.value)
        return priority_value_list

    def is_anyone_in_queue(self):
        return True if len(self.agents_in_queue) != 0 else False

    def get_no_of_prior_agents_in_queue(self):
        counter = 0
        for agent in self.agents_in_queue:
            if agent.is_prioritized:
                counter += 1
        return counter

    def get_est_to_finish_excluding_agents_time(self):
        remaining_time = 0

        if self.is_anyone_in_service():
            agent_in_service = self.agents_in_service[0]
            remaining_time += agent_in_service.get_task(TaskStatus.PROCESSING).finish_time - env.current_time

        if self.is_anyone_in_queue():
            remaining_time += self.job_duration * len(self.agents_in_queue)

        return remaining_time

    def get_est_time_to_wait_for_an_agent(self, agent: Agent):
        waiting_time = 0
        if self.is_anyone_in_service():
            agent_in_service = self.agents_in_service[0]
            waiting_time += agent_in_service.get_task(TaskStatus.PROCESSING).finish_time - env.current_time

        if agent in self.agents_in_queue:
            agent_position = self.agents_in_queue.index(agent)
            waiting_time += agent_position * self.job_duration
        else:
            waiting_time += len(self.agents_in_queue) * self.job_duration

        return waiting_time

    def get_estimated_finish_time_for_specific_agent_by_state_vector(self, state_vector, this_agent):
        estimated_finish_time = 0
        if this_agent is None:
            estimated_finish_time = self.get_queue_size() * self.job_duration + state_vector[2][1]
        else:
            num_of_hogher_priority = 0
            for others_priority in state_vector[4]:
                if this_agent.priority.value >= others_priority:
                    num_of_hogher_priority += 1
            estimated_finish_time = num_of_hogher_priority * self.job_duration + state_vector[2][1]

        return estimated_finish_time

    def get_machine_state(self, current_time=0, check_for_this_agent: Agent = None):
        """
        [
            0 - machine id,
            1- count (Queue) ,
            2- [
                num_of_agent_in_service,
                remaining time for the current agent is serving
            ],
            3- count (Num of Priorities),
            4- [priority values],
            5 -count (Finished tasks),
            6- count (Estimated time to finish all the jobs)
        ]
        """
        state = []
        # machine id
        state.insert(0, self.machine_id)
        # num of queue
        state.insert(1, len(self.agents_in_queue))

        # [
        #       num_of_agent_in_service,
        #       remaining time for the current agent is serving
        # ]

        state.insert(2, [1 if self.is_anyone_in_service() else 0,
                         self.get_remaining_time_for_serving_agent(current_time)])

        # num of prior agent
        state.insert(3, self.get_no_of_prior_agents_in_queue())
        # priority values
        state.insert(4, self.get_priority_values_in_the_queue())

        state.insert(5, len(self.agents_finished))

        state.insert(6, self.get_estimated_finish_time_for_specific_agent_by_state_vector(state, check_for_this_agent))
        return state

    def print_machine_info(self, with_finish=False):
        msg = f"\n\t{self.name} - machine status:"
        msg += f"\n\t\tQUEUE({len(self.agents_in_queue)}): {[agent.id_with_prior_info() for agent in self.agents_in_queue]}"
        msg += f"\n\t\tSERVING({len(self.agents_in_service)}): {[agent.id_with_prior_info() for agent in self.agents_in_service]}"

        if with_finish:
            msg += f"\n\t\tFINISHED({len(self.agents_finished)}): {[agent.id_with_prior_info() for agent in self.agents_finished]}"

        print(msg)

    def put_agent_in_queue(self, agent: Agent):
        def find_position_to_insert_prior_agent(queue, priority):
            if len(queue) == 0:
                return 0
            position = len(queue) - 1
            pos = 0
            while position >= 0:
                if not queue[position].is_prioritized:
                    pos = position
                else:
                    if queue[position].priority.value < priority.value:
                        pos = position
                    elif queue[position].priority.value == priority.value:
                        pos = position
                        break

                position -= 1

            return pos if pos >= 0 else 0

        if not agent.is_prioritized:
            self.agents_in_queue.append(agent)
        else:
            pos = find_position_to_insert_prior_agent(self.agents_in_queue, agent.priority)
            if self.get_queue_size() > 0:
                if self.agents_in_queue[pos].priority.value == agent.priority.value:
                    pos += 1
            self.agents_in_queue.insert(pos, agent)

        #self.print_machine_info()

    def get_the_serving_agent(self):
        return self.agents_in_service[0]

    def serve_the_next_agent(self, time):
        if not self.is_anyone_in_queue() or self.is_anyone_in_service():
            return

        agent_to_serve: Agent = self.agents_in_queue.pop(0)
        agent_to_serve.serving_time = time
        agent_to_serve.waiting_time = agent_to_serve.serving_time - agent_to_serve.arrival_time
        finish_time = agent_to_serve.serving_time + self.job_duration
        agent_to_serve.finish_time = finish_time

        current_task: Task = agent_to_serve.get_task(TaskStatus.QUEUE)
        current_task.status = TaskStatus.PROCESSING
        current_task.update_timing_info(serving_time=time,
                                        finishing_time=finish_time,
                                        waiting_time=time - current_task.arrival_time)
        change_next_task: Task = agent_to_serve.get_task(TaskStatus.NOTREADY)
        if change_next_task != None:

            change_next_task.status = TaskStatus.GENERATED

        if finish_time > SimParams.SIM_TIME:
            SimParams.SIM_TIME = finish_time

        self.agents_in_service.append(agent_to_serve)
        #print(f"{agent_to_serve.status_message()} \n\tSERVING in {self.name}")
        #self.print_machine_info()

    def finish_this_agents_job(self):
        agent_that_served: Agent = self.agents_in_service.pop(0)
        self.agents_finished.append(agent_that_served)
        #self.print_machine_info(with_finish=True)


class Sanding(Machine):
    def __init__(self, machine_id, name, duration=0):
        super(Sanding, self).__init__(machine_id, name, MachineType.SANDING)
        self.name = name
        self.job_duration = duration


class Drill(Machine):
    def __init__(self, machine_id, name, duration=0):
        super(Drill, self).__init__(machine_id, name, MachineType.DRILL)
        self.job_duration = duration


class Paint(Machine):
    def __init__(self, machine_id, name, duration=0):
        super(Paint, self).__init__(machine_id, name, MachineType.PAINT)
        self.job_duration = duration

# only two type of machine
class env:
    machine_collection = {
        MachineType.SANDING: [
            Sanding(1, "SANDING-1", duration=4),
            Sanding(2, "SANDING-2", duration=6)
        ],

        MachineType.DRILL: [
            Drill(3, "DRILL-1", duration=4),
            Drill(4, "DRILL-2", duration=7)

        ]

    }

    """
    machine_collection = {
        MachineType.SANDING: [
            Sanding(1, "SANDING-1"),
            Sanding(2, "SANDING-2")
        ],
        MachineType.DRILL: [
            Drill(3, "DRILL-1"),
            Drill(4, "DRILL-2"),
            Drill(5, "DRILL-3")
        ],
        MachineType.PAINT: [
            Paint(6, "PAINT-1")
        ]
    }
    """

    future_agents = {}
    all_agents = []
    current_agents = []
    current_time = 0
    step_penalty   = -1
    success_reward = 100
    _instance = None

    @classmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_machine_by_type(cls, type):
        return env.machine_collection[type]

    @classmethod
    def exp_episode_to_csv(cls, episode_num, num_of_machines_per_cat = 2):

        '''
       episode_num, agent-1, agent-2 ...    agent-10
       1            1,      1
       1            3,      3



       {
       "1":{
               "0": [1,3],
               "1": [1,3]
           }
        }

        [1, 1]
        '''

        data = {
            episode_num: {}
        }

        for _, machines in env.machine_collection.items():  # SANDING, DRILLING
            for machine in machines:  # SANDING-1...DRILL-2
                for agent in machine.agents_finished:  # SANDING-1 [1,2...]
                    if agent.agent_id not in data[episode_num]:
                        data[episode_num][agent.agent_id] = []
                    data[episode_num][agent.agent_id].append(machine.machine_id)

        index = 0 # 1
        row = []  # [1, 3, 3]

        while num_of_machines_per_cat > 0:
            row.insert(0, episode_num)
            for key, val in data[episode_num].items():

                try:
                    row.insert(int(key) + 1, val[index])
                except:
                    row.insert(int(key) + 1,  -1)

            ToCSV.publish_row(row)
            row.clear()
            index = index + 1
            num_of_machines_per_cat = num_of_machines_per_cat - 1

    class SimStateRunner:

        @staticmethod
        def bring_agents_in_queue(current_time, future_agents, show_printouts):
            if current_time in future_agents:
                for agent in future_agents[current_time]:
                    next_task_to_execute: Task = agent.get_next_task(current_time, show_printouts)
                    next_task_to_execute.update_timing_info(arrival_time=current_time)
                    if show_printouts:
                        print(f"⌚ {current_time}: {agent.status_message()} "
                              f"\n\tARRIVED in {next_task_to_execute.machine.name}")
                    next_task_to_execute.machine.put_agent_in_queue(agent)

        @staticmethod
        def bring_turn_based_agents_in_queue(current_time, future_agents, agent, input_machine, show_printouts):

            if current_time in future_agents:
                future_agents_ids = [a.agent_id for a in future_agents[current_time]]
                if agent.agent_id in future_agents_ids:
                    next_task_to_execute: Task = agent.action(current_time, input_machine)
                    next_task_to_execute.update_timing_info(arrival_time=current_time)
                    if show_printouts:
                        print(f"⌚ {current_time}: {agent.status_message()} "
                              f"\n\tARRIVED in {next_task_to_execute.machine.name}")
                    next_task_to_execute.machine.put_agent_in_queue(agent)
                    env.current_agents.append(agent)

        @staticmethod
        def get_legal_moves(id, current_time, future_agents, current_agents, all_machines):

            future_agents_ids = []
            current_agents_ids = [a.agent_id for a in current_agents]
            # If agent_id cannot make a move or is not ready yet, return [0]
            if current_time in future_agents:
                future_agents_ids = [a.agent_id for a in future_agents[current_time]]
            if id not in future_agents_ids and id not in current_agents_ids:
                return [0]

            # Extract agent. This is important because it may be out of order on creation
            agent = [agent for agent in env.all_agents if agent.agent_id == id][0]

            # If agent is currently in service, return action is 0
            if agent.get_task(TaskStatus.PROCESSING) and agent.finish_time > current_time + 1:
                return [0]

            possible_machines = []

            for task in agent.task_list:

                if task.status == TaskStatus.GENERATED:
                    for machine in task.machine_lists_of_this_type:
                        possible_machines.append(machine.get_machine_state(current_time)[0])
            if not possible_machines:
                return [0]

            return possible_machines

            # possible_machines = []
            #
            # # possible_machines = [0]
            # for task in agent.task_list:
            #     if task.machine_type == MachineType.SANDING and agent.agent_id == 0 and task.status == TaskStatus.GENERATED:
            #         possible_machines.append(1)
            #     elif task.machine_type == MachineType.DRILL and agent.agent_id == 0 and task.status == TaskStatus.GENERATED:
            #         possible_machines.append(3)
            #
            #     elif task.status == TaskStatus.GENERATED:
            #         for machine in task.machine_lists_of_this_type:
            #             possible_machines.append(machine.get_machine_state(current_time)[0])
            # if not possible_machines:
            #     return [0]
            #
            # return possible_machines



        @staticmethod
        def serve_agents(current_time, all_machines):
            for m_type, machine_list in all_machines.items():
                for machine in machine_list:
                    machine.serve_the_next_agent(current_time)

        @staticmethod
        def finish_turn_based_agents_job(current_time, all_machines, id, input_machine, show_printouts):

            for m_type, machine_list in all_machines.items():
                for machine in machine_list:
                    if machine.is_anyone_in_service():
                        agent = machine.get_the_serving_agent()
                        # Check if current player matches ID
                        if agent.agent_id == id and agent.finish_time == current_time:
                            task: Task = agent.get_task(TaskStatus.PROCESSING)
                            task.update_timing_info(finishing_time=current_time)
                            task.status = TaskStatus.FINISH
                            if show_printouts:
                                print(f"{agent.status_message()} \n\tFINISHED in {machine.name}")
                            machine.finish_this_agents_job()
                            # Replaced get_next_task with action function
                            next_task_for_this_agent: Task = agent.action(current_time, input_machine)
                            if next_task_for_this_agent:
                                agent.arrival_time = current_time
                                next_task_for_this_agent.update_timing_info(arrival_time=current_time)
                                if show_printouts:
                                    print(f"{agent.status_message()} \n\tARRIVED in "
                                          f"{next_task_for_this_agent.machine.name}")
                                next_task_for_this_agent.machine.put_agent_in_queue(agent)
                                if len(next_task_for_this_agent.machine.agents_in_queue) == 1:
                                    next_task_for_this_agent.machine.serve_the_next_agent(current_time)

                            # Trigger the next agent, from que to service for this machine
                            machine.serve_the_next_agent(current_time)
                        if agent.has_all_task_finished():
                            SimParams.ALL_FINISHED_AGENTS.append(agent)

        @staticmethod
        def finish_agents_job(current_time, all_machines, show_printouts):
            for m_type, machine_list in all_machines.items():
                for machine in machine_list:
                    if machine.is_anyone_in_service():
                        agent = machine.get_the_serving_agent()
                        if agent.finish_time == current_time:
                            task: Task = agent.get_task(TaskStatus.PROCESSING)
                            task.update_timing_info(finishing_time=current_time)
                            task.status = TaskStatus.FINISH
                            if show_printouts:
                                print(f"{agent.status_message()} \n\tFINISHED in {machine.name}")
                            machine.finish_this_agents_job()
                            next_task_for_this_agent: Task = agent.get_next_task(current_time, show_printouts)

                            if next_task_for_this_agent:
                                agent.arrival_time = current_time
                                next_task_for_this_agent.update_timing_info(arrival_time=current_time)
                                if show_printouts:
                                    print(f"{agent.status_message()} \n\tARRIVED in "
                                          f"{next_task_for_this_agent.machine.name}")
                                next_task_for_this_agent.machine.put_agent_in_queue(agent)
                                if len(next_task_for_this_agent.machine.agents_in_queue) == 1:
                                    next_task_for_this_agent.machine.serve_the_next_agent(current_time)

                            # Trigger the next agent, from que to service for this machine
                            machine.serve_the_next_agent(current_time)
                        if agent.has_all_task_finished():
                            SimParams.ALL_FINISHED_AGENTS.append(agent)

    @staticmethod
    def write_to_file(path):
        GenerateSimData.write_to_file(path)

    @staticmethod
    def add_agent(time, new_generated_agent):
        if time not in env.future_agents:
            env.future_agents[time] = []
        env.future_agents[time].append(new_generated_agent)
        print(env.future_agents[time])

    @staticmethod
    def generate_agent():
        time = 0
        agent_counter = 0
        print("num of order", SimParams.N_ORDER)
        while agent_counter < SimParams.N_ORDER:

            num_of_agent_to_create = random.randint(1, SimParams.MAX_NUM_OF_AGENT_GEN)

            if agent_counter + num_of_agent_to_create > SimParams.N_ORDER:
                num_of_agent_to_create = (SimParams.N_ORDER - agent_counter)

            for index in range(num_of_agent_to_create):
                next_agent_arrival_time = time + \
                                          random.randint(SimParams.AGN_GEN_INTRVL_MIN,
                                                         SimParams.AGN_GEN_INTRVL_MAX) if time != 0 else time

                new_agent = Agent(SimParams.AGENT_ID, next_agent_arrival_time, env.machine_collection)
                env.all_agents.append(new_agent)
                SimParams.AGENT_ID += 1
                env.add_agent(next_agent_arrival_time, new_agent)

                agent_counter += 1

            time += 1

    @staticmethod
    def generate_agent_fixed_Size():
        a1 = Agent(0, 0, env.machine_collection)
        env.all_agents.append(a1)
        env.add_agent(0, a1)

        a2 = Agent(1, 0, env.machine_collection)
        env.add_agent(0, a2)
        env.all_agents.append(a2)

        #a3 = Agent(2, 0, env.machine_collection)
        #env.add_agent(0, a3)
        #env.all_agents.append(a3)
        # #
        # a4 = Agent(3, 4, env.machine_collection)
        # env.add_agent(4, a4)
        # env.all_agents.append(a4)
        #
        # a5 = Agent(4, 4, env.machine_collection)
        # env.add_agent(4, a5)
        # env.all_agents.append(a5)

    """
    Get all observation vectors from all agents
    """

    @classmethod
    def get_all_observation_vectors(cls):
        def get_type_wise_machine_list():
            machine_id_list = []
            machine_type_list = []

            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_id_list.append(machine.machine_id)
                    machine_type_list.append(MachineType.get_type_index(machine.machine_type))

            return [machine_id_list, machine_type_list]

        '''
        def get_machine_id_list():
            machine_id_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_id_list.append(machine.machine_id)
            return machine_id_list
        '''

        def get_machine_usage_status_list():
            machine_usage_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_usage_list.append(1 if machine.is_anyone_in_service() else 0)
            return machine_usage_list

        def get_machine_remaining_time_for_specific_agent(current_agent: Agent):
            remaining_time_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    time = machine.get_est_time_to_wait_for_an_agent(current_agent)
                    remaining_time_list.append(time)

            return remaining_time_list

        def get_machines_queue_positions():
            queue_position_list = []
            num_of_agents = len(env.all_agents)

            for machine_type, machines in env.machine_collection.items():
                current_machines_queue = [0] * num_of_agents
                for machine in machines:
                    for idx in range(len(machine.agents_in_queue)):
                        current_machines_queue[idx] = 1
                    queue_position_list.append(current_machines_queue)

            return queue_position_list

        def get_machines_priority_position_list():
            priority_info_list = []
            num_of_agents = len(env.all_agents)

            for machine_type, machines in env.machine_collection.items():
                current_machine_queue = [0] * num_of_agents
                for machine in machines:
                    for idx in range(len(machine.agents_in_queue)):
                        agent = machine.agents_in_queue[idx]
                        current_machine_queue[idx] = (1 if agent.is_prioritized else 0)
                    priority_info_list.append(current_machine_queue)

            return priority_info_list

        def get_agent_current_position_data(current_agent: Agent):
            current_task = current_agent.get_task(TaskStatus.QUEUE)

            if current_task is None:
                current_task = current_agent.get_task(TaskStatus.PROCESSING)
                if current_task:
                    return [
                        current_task.machine.machine_id,
                        0,
                        1 if current_agent.is_prioritized else 0,
                        1
                    ]

            if current_task is None:
                return [-1, -1, -1, -1]
            return [
                current_task.machine.machine_id,
                current_task.machine.agents_in_queue.index(current_agent),
                1 if current_agent.is_prioritized else 0,
                0
            ]

        def get_agents_finished_task_status_list(current_agent: Agent):
            # [unserved, currently_served,finished]
            if current_agent.has_served_by_any_machine():
                machine_involved_for_this_agent = []
                agent_task_finish_status_list = []
                for task in current_agent.task_list:
                    machine_id = -1
                    status = -1
                    if task.machine:
                        machine_id = task.machine.machine_id
                        status = 1 if task.status == TaskStatus.FINISH else 0
                    machine_involved_for_this_agent.append(machine_id)
                    agent_task_finish_status_list.append(status)
                return [agent_task_finish_status_list.count(-1), agent_task_finish_status_list.count(0),
                        agent_task_finish_status_list.count(1)]
                # return [machine_involved_for_this_agent, agent_task_finish_status_list]

            return [len(current_agent.task_list), 0, 0]
            # return [[-1] * len(current_agent.task_list), [-1] * len(current_agent.task_list)]

        def populate_vector(agent_list):
            all_agent_vector = []
            for agent in agent_list:
                single_agent_vector = []

                single_agent_vector.append(get_type_wise_machine_list())
                # single_agent_vector.append(get_machine_id_list())
                single_agent_vector.append(get_machine_usage_status_list())
                single_agent_vector.append(get_machine_remaining_time_for_specific_agent(agent))
                single_agent_vector.append(get_machines_queue_positions())
                single_agent_vector.append(get_machines_priority_position_list())
                single_agent_vector.append(get_agent_current_position_data(agent))
                single_agent_vector.append(get_agents_finished_task_status_list(agent))
                single_agent_vector.append(agent.agent_id)

                all_agent_vector.append(single_agent_vector)

            return all_agent_vector

        agent_vector = []
        for data in populate_vector(cls.all_agents):
            agent_vector.append(data)
        return agent_vector

    @classmethod
    def get_observation_vector(cls):

        def get_type_wise_machine_list():
            machine_id_list = []
            machine_type_list = []

            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_id_list.append(machine.machine_id)
                    machine_type_list.append(MachineType.get_type_index(machine.machine_type))

            return [machine_id_list, machine_type_list]

        '''
        def get_machine_id_list():
            machine_id_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_id_list.append(machine.machine_id)
            return machine_id_list
        '''

        def get_machine_usage_status_list():
            machine_usage_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    machine_usage_list.append(1 if machine.is_anyone_in_service() else 0)
            return machine_usage_list

        def get_machine_remaining_time_for_specific_agent(current_agent: Agent):
            remaining_time_list = []
            for machine_type, machines in env.machine_collection.items():
                for machine in machines:
                    time = machine.get_est_time_to_wait_for_an_agent(current_agent)
                    remaining_time_list.append(time)

            return remaining_time_list




        def get_machines_queue_positions():
            queue_position_list = []

            num_of_agents = len(env.all_agents)


            for machine_type, machines in env.machine_collection.items():
                #current_machines_queue = [0] * num_of_agents
                for machine in machines:
                    current_machines_queue = [0] * num_of_agents
                    for idx in range(len(machine.agents_in_queue)):
                        current_machines_queue[idx] = 1

                    #current_machines_queue
                    queue_position_list.append(current_machines_queue)
                    #current_machines_queue.clear()
                    #current_machines_queue = [0] * num_of_agents




            return queue_position_list

        def get_machines_priority_position_list():
            priority_info_list = []
            num_of_agents = len(env.all_agents)

            for machine_type, machines in env.machine_collection.items():

                for machine in machines:
                    current_machine_queue = [0] * num_of_agents
                    for idx in range(len(machine.agents_in_queue)):
                        agent = machine.agents_in_queue[idx]
                        current_machine_queue[idx] = (1 if agent.is_prioritized else 0)
                    priority_info_list.append(current_machine_queue)

            return priority_info_list

        def get_agent_current_position_data(current_agent: Agent):
            current_task = current_agent.get_task(TaskStatus.QUEUE)

            if current_task is None:
                current_task = current_agent.get_task(TaskStatus.PROCESSING)
                if current_task:
                    return [
                        current_task.machine.machine_id,
                        0,
                        1 if current_agent.is_prioritized else 0,
                        1
                    ]

            if current_task is None:
                return [-1, -1, -1, -1]
            return [
                current_task.machine.machine_id,
                current_task.machine.agents_in_queue.index(current_agent),
                1 if current_agent.is_prioritized else 0,
                0
            ]

        def get_agents_finished_task_status_list(current_agent: Agent):
            if current_agent.has_served_by_any_machine():
                machine_involved_for_this_agent = []
                agent_task_finish_status_list = []
                for task in current_agent.task_list:
                    machine_id = -1
                    status = -1
                    if task.machine:
                        machine_id = task.machine.machine_id
                        status = 1 if task.status == TaskStatus.FINISH else 0
                    machine_involved_for_this_agent.append(machine_id)
                    agent_task_finish_status_list.append(status)
                return [agent_task_finish_status_list.count(-1), agent_task_finish_status_list.count(0),
                        agent_task_finish_status_list.count(1)]
                # return [machine_involved_for_this_agent, agent_task_finish_status_list]

            return [len(current_agent.task_list), 0, 0]
            # return [[-1] * len(current_agent.task_list), [-1] * len(current_agent.task_list)]

        def populate_vector(agent_list):
            all_agent_vector = []
            for agent in agent_list:
                single_agent_vector = []

                single_agent_vector.append(get_type_wise_machine_list())
                # single_agent_vector.append(get_machine_id_list())
                single_agent_vector.append(get_machine_usage_status_list())
                single_agent_vector.append(get_machine_remaining_time_for_specific_agent(agent))
                single_agent_vector.append(get_machines_queue_positions())
                single_agent_vector.append(get_machines_priority_position_list())
                single_agent_vector.append(get_agent_current_position_data(agent))
                single_agent_vector.append(get_agents_finished_task_status_list(agent))
                single_agent_vector.append(agent.agent_id)

                all_agent_vector.append(single_agent_vector)

            return all_agent_vector

        agent_vector = []

        # For agents generated in current time,  those are not assigned to any machine
        if cls.current_time in cls.future_agents:
            # for newly generated agents
            for data in populate_vector(cls.future_agents[cls.current_time]):
                agent_vector.append(data)

        # For agents in the queue and serving
        for m_type, all_machines in env.machine_collection.items():
            for machine in all_machines:
                # for agent in queue
                for data in populate_vector(machine.agents_in_queue):
                    agent_vector.append(data)

                # for agent in serving
                if machine.is_anyone_in_service():
                    for data in populate_vector(machine.agents_in_service):
                        agent_vector.append(data)
        return agent_vector

    @staticmethod
    def print_observation_vector(vector):
        print("\n\n\t Agent Vector:")
        for item in vector:
            print(f"\n\t\tagent id: {item[7]}")
            print(f"\t\tMachine with type: {item[0]}")
            # print(f"\t\tMachine ids: {item[1]}")
            print(f"\t\tMachine occupied?: {item[1]}")
            print(f"\t\tRemaining time for specific agent : {item[2]}")
            print("\t\tMachine queue positions:")
            for m in item[3]:
                print(f"\t\t\t{m}")
            print("\t\tMachine priority position in queue:")
            for m in item[4]:
                print(f"\t\t\t{m}")
            print(f"\t\tAgents positional data: {item[5]}")
            print(f"\t\tAgents finished tasks data: {item[6]}")
            print("\t\t========================================================")

    # Info state takes observation vector from all observations and flattens it into single input array for each agent
    @classmethod
    def get_info_state(cls, agent_id):
        observations = env.get_all_observation_vectors()[agent_id]
        flattened_observation_vector = []
        for obs in observations:
            flattened_observation_vector = np.concatenate((flattened_observation_vector, np.array(obs).flatten()),
                                                          axis=None)
        return flattened_observation_vector

    @classmethod
    def get_info_state_shape(cls):
        observations = env.get_all_observation_vectors()[0]
        flattened_observation_vector = []
        for obs in observations:
            flattened_observation_vector = np.concatenate((flattened_observation_vector, np.array(obs).flatten()),
                                                          axis=None)
        return flattened_observation_vector.shape[0]

    # Turn Based Step
    @classmethod
    def turn_based_step(cls, current_time, action_list, legal_actions, show_printouts=True):

        cls.current_time = current_time // SimParams.N_ORDER

        current_agent_id = current_time % SimParams.N_ORDER

        agent = [a for a in cls.all_agents if a.agent_id == current_agent_id][0]
        input_machine = action_list[0]

        # Generate agent vector
        if show_printouts and input_machine != 0:
            env.print_observation_vector(env.get_observation_vector())

        # Process queue
        env.SimStateRunner.bring_turn_based_agents_in_queue(cls.current_time, cls.future_agents, agent, input_machine,
                                                            show_printouts)

        queue_machine = 0
        for _, machines in env.machine_collection.items():
            for i, machine in enumerate(machines):
                if agent in machine.agents_in_queue:
                    queue_machine = machine.machine_id

        # # Queue switching
        # # If action is not 0 and agent has a task that is processing and it's going to a new queue
        # if agent.get_task(
        #         TaskStatus.QUEUE) and input_machine != 0 and queue_machine != 0 and queue_machine != input_machine:
        #     # Reset current task
        #     task: Task = agent.get_task(TaskStatus.QUEUE)
        #     task.update_timing_info(finishing_time=-1)
        #     task.status = TaskStatus.GENERATED
        #     task.machine.agents_in_queue.remove(agent)
        #     task.machine = None
        #
        #     # Go to new machine
        #     next_task_for_this_agent: Task = agent.action(current_time, input_machine)
        #     if next_task_for_this_agent:
        #         agent.arrival_time = current_time
        #         next_task_for_this_agent.update_timing_info(arrival_time=current_time)
        #         next_task_for_this_agent.machine.put_agent_in_queue(agent)
        # elif input_machine != 0 and agent in env.current_agents and agent.get_task(
        #         TaskStatus.GENERATED) and not agent.get_task(TaskStatus.QUEUE) and not agent.get_task(
        #     TaskStatus.PROCESSING):
        #     next_task_to_execute: Task = agent.action(current_time, input_machine)
        #     next_task_to_execute.machine.put_agent_in_queue(agent)

        # If in last timestep, handles serving
        if current_agent_id == SimParams.N_ORDER - 1:
            # Handle serving
            env.SimStateRunner.serve_agents(cls.current_time, env.machine_collection)
            # print(f"FINISHED: {len(SimParams.ALL_FINISHED_AGENTS)}")

        # Handling finish
        env.SimStateRunner.finish_turn_based_agents_job(
            cls.current_time, env.machine_collection, current_agent_id, input_machine, show_printouts)

        if current_agent_id == SimParams.N_ORDER - 1:
            # Generating json of the sim data
            GenerateSimData.generate_time_specific_data(cls.current_time, env.machine_collection)

        # Update TimeStep environment
        #if  legal_actions:
        #if legal_actions[0] != 0:
        observations = {
            "info_state": [],
            "legal_actions": [],
            "current_player": [],
            "serialized_state": []
        }
        for id in range(SimParams.N_ORDER):
            # If conditation
            observations["info_state"].append(env.get_info_state(id))
            observations["legal_actions"].append(env.SimStateRunner.get_legal_moves(
                id, cls.current_time, cls.future_agents, cls.current_agents, env.machine_collection.items()))
        next_player = (current_time + 1) % SimParams.N_ORDER
        observations["current_player"] = next_player
        # Last timestep
        if len(SimParams.ALL_FINISHED_AGENTS) == SimParams.N_ORDER:
            # If last unit and last timestep
            if current_agent_id == SimParams.N_ORDER - 1:
                return TimeStep(
                    observations=observations,
                    rewards=[env.success_reward for i in range(SimParams.N_ORDER)],
                    discounts=[0.99 for i in range(SimParams.N_ORDER)],
                    step_type=StepType.LAST)
            else:
                return TimeStep(
                    observations=observations,
                    rewards=[env.step_penalty for i in range(SimParams.N_ORDER)],
                    discounts=[0.99 for i in range(SimParams.N_ORDER)],
                    step_type=StepType.MID)

        # -1 reward at each timestep, optimizes least amount of timesteps
        # Discount set to 0.99, future rewards are worth 99% of prior rewards, tuneable
        return TimeStep(
            observations=observations,
            rewards=[env.step_penalty for i in range(SimParams.N_ORDER)],
            discounts=[0.99 for i in range(SimParams.N_ORDER)],
            step_type=StepType.MID)

    @classmethod
    def reset(cls, num_of_orders, show_printouts,step_penalty = -1, success_reward = 100):
        SimParams.reset()
        GenerateSimData.SIM_DATA = {}

        for _, machines in env.machine_collection.items():
            for machine in machines:
                machine.reset()

        env.future_agents.clear()
        env.all_agents.clear()
        env.current_time = 0
        env.step_penalty = step_penalty
        env.success_reward = success_reward
        
        # Moved starter code into reset
        SimParams.N_ORDER = num_of_orders
        # env.generate_agent()
        # TODO: Change
        env.generate_agent_fixed_Size()

        if show_printouts:
            env.print_info()

        GenerateSimData.generate_machine_info()
        # GenerateSimData.generate_agent_info(env.future_agents)

        observations = {
            "info_state": [],
            "legal_actions": [],
            "current_player": [],
            "serialized_state": []
        }

        # Turn Base Action
        for id in range(num_of_orders):
            observations["info_state"].append(env.get_info_state(id))
            observations["legal_actions"].append(env.SimStateRunner.get_legal_moves(
                id, env.current_time, env.future_agents, env.current_agents, env.machine_collection.items()))
        observations["current_player"] = 0

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    @classmethod
    def run(cls, num_of_orders=0, show_printouts=False):
        SimParams.N_ORDER = num_of_orders
        # env.generate_agent()
        # TODO Change
        env.generate_agent_fixed_Size()
        # env.generate_agent(machines)
        if show_printouts:
            env.print_info()
            GenerateSimData.generate_agent_info(env.future_agents)

        while cls.current_time <= SimParams.SIM_TIME:
            # Generate agent vector
            if show_printouts:
                print(f"⌚ {cls.current_time}:\n")
                env.print_observation_vector(env.get_observation_vector())

            # Process queue
            # Action space inputs
            env.SimStateRunner.bring_agents_in_queue(cls.current_time, cls.future_agents, show_printouts)

            # Handle serving
            env.SimStateRunner.serve_agents(cls.current_time, env.machine_collection)
            # Handling finish
            env.SimStateRunner.finish_agents_job(cls.current_time, env.machine_collection, show_printouts)

            # Generating json of the sim data
            # GenerateSimData.generate_time_specific_data(cls.current_time, env.machine_collection)
            cls.current_time += 1
            if show_printouts:
                print("--" * 40)
        if show_printouts:
            GenerateSimData.generate_task_status(cls.all_agents)
        print(f"It took {cls.current_time - 1} unit of time to complete {SimParams.N_ORDER} agents !! Bye Bye :) !!")
        return cls.current_time - 1

    @classmethod
    def reset_old(cls):
        SimParams.reset()
        for _, machines in env.machine_collection.items():
            for machine in machines:
                machine.reset()
        env.future_agents.clear()
        env.all_agents.clear()
        env.current_time = 0

    @classmethod
    def print_info(cls):
        print("AT = Arrival time, ST = Service time, WT = Waiting time, FT = Finishing time.\n")
        print(""" Machine state
                [
                    0 - machine id,
                    1- count (Queue) ,
                    2- [
                        num_of_agent_in_service,
                        remaining time for the serving agent to finish
                    ],
                    3- count (Num of Priorities),
                    4- [priority flags of the priority agents in queue],
                    5- count (Finished tasks),
                    6- count (Estimated time to finish all the jobs)
                ]
                """)


class TimeStep(
    collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])):
    __slots__ = ()

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def is_simultaneous_move(self):
        return False
        # return self.observations["current_player"] == SIMULTANEOUS_PLAYER_ID

    def current_player(self):
        return self.observations["current_player"]


class StepType(Enum):
    FIRST = 0
    MID = 1
    LAST = 2

    def first(self):
        return self is StepType.FIRST

    def mid(self):
        return self is StepType.MID

    def last(self):
        return self is StepType.LAST


class ToCSV:
    _instance = None
    file_name = "agent_history_episode_wise.csv"
    headers = []

    @classmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    @classmethod
    def initiate(cls, columns, filepath=None):
        ToCSV.headers = columns
        if filepath is not None:
            ToCSV.file_name = filepath + "/" + ToCSV.file_name

        with open(ToCSV.file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ToCSV.headers)
        f.close()

    @classmethod
    def publish_row(cls, row):
        with open(ToCSV.file_name, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        f.close()
