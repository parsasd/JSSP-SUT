from yafs.application import Application, Message
from yafs.distribution import deterministic_distribution

class JSSPWorkload:
    def __init__(self, raw_jobs):
        self.raw_jobs = raw_jobs  # Dict: {job_id: [{'machine': m, 'duration': d}, ...]}

    def create_application(self, job_release_times):
        """
        Creates a SINGLE Application representing the entire Job Shop.
        - Modules: Machine_0, Machine_1, ... (Shared Resources)
        - Sources: Job_0_Start, Job_1_Start ... (Job Triggers)
        """
        app_name = "JSSP_Shop"
        app = Application(name=app_name)
        
        # 1. Identify all unique machines to prep the Placement Map
        # We DO NOT register dummy services here anymore (this caused the crash).
        # We just explicitly track which module name maps to which machine ID.
        if not hasattr(app, "machine_map"): 
            app.machine_map = {}

        all_machines = set()
        for ops in self.raw_jobs.values():
            for op in ops:
                all_machines.add(op['machine'])
                
        for m_id in all_machines:
            module_name = f"Machine_{m_id}"
            app.machine_map[module_name] = m_id

        # 2. Create Job Sources and Wiring
        for job_id, operations in self.raw_jobs.items():
            # A. Source
            source_name = f"Source_Job_{job_id}"
            start_time = job_release_times.get(job_id, 0)
            
            # The first message goes to the first machine in the job's sequence
            first_op = operations[0]
            first_machine = f"Machine_{first_op['machine']}"
            
            # Create the initial message
            # Note: instructions = duration. We assume Node IPT = 1.
            msg_start_name = f"Msg_J{job_id}_Start"
            msg_start = Message(msg_start_name, source_name, first_machine, 
                                instructions=first_op['duration'], bytes=100)
            msg_start.last_idDes = []  # Required by YAFS patch
            app.messages[msg_start_name] = msg_start

            # Deploy the source
            dist = deterministic_distribution(name=f"Start_Dist_J{job_id}", time=start_time)
            app.add_service_source(source_name, dist, msg_start)

            # B. Chain Operations (The Route)
            # We route from Machine A -> Machine B -> ... -> Sink
            for i in range(len(operations)):
                current_op = operations[i]
                current_machine_name = f"Machine_{current_op['machine']}"
                
                # Determine Next Step
                if i < len(operations) - 1:
                    next_op = operations[i+1]
                    next_machine_name = f"Machine_{next_op['machine']}"
                    
                    # Create Message for Next Hop
                    msg_next_name = f"Msg_J{job_id}_Op{i}_to_Op{i+1}"
                    msg_next = Message(msg_next_name, current_machine_name, next_machine_name,
                                       instructions=next_op['duration'], bytes=100)
                    msg_next.last_idDes = []
                    app.messages[msg_next_name] = msg_next
                    
                    # Register this specific transition on the Shared Machine Service
                    # Input: The message arriving from Prev (or Source)
                    # Output: The message going to Next
                    if i == 0:
                        msg_in = msg_start
                    else:
                        prev_op = operations[i-1]
                        prev_machine = f"Machine_{prev_op['machine']}"
                        # Reconstruct the name of the incoming message
                        msg_in_name = f"Msg_J{job_id}_Op{i-1}_to_Op{i}"
                        msg_in = app.messages[msg_in_name]

                    # Add the service rule for this specific job step
                    app.add_service_module(
                        current_machine_name,
                        message_in=msg_in,
                        message_out=msg_next,
                        distribution=lambda **_: True, # Pass-through
                        module_dest=[next_machine_name],
                        p=[1.0]
                    )

                else:
                    # Last Operation: Sink (No output message)
                    prev_op = operations[i-1]
                    msg_in_name = f"Msg_J{job_id}_Op{i-1}_to_Op{i}"
                    msg_in = app.messages[msg_in_name]
                    
                    app.add_service_module(
                        current_machine_name,
                        message_in=msg_in,
                        message_out=None, # SINK
                        distribution=lambda **_: True,
                        module_dest=[],
                        p=[]
                    )

        return app