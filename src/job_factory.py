from typing import Dict, List, Any

from yafs.application import Application, Message
from yafs.distribution import deterministic_distribution


class JSSPWorkload:
    def __init__(self, raw_jobs: Dict[int, List[Dict[str, Any]]], operation_specs: List[Dict[str, Any]]):
        """
        raw_jobs: canonical operations (duration, machine id) per job
        operation_specs: flattened list of operations with eligible nodes
        """
        self.raw_jobs = raw_jobs
        self.operation_specs = operation_specs

    def create_application(self, job_release_times, module_node_map):
        """
        Creates a SINGLE Application representing the entire Job Shop.
        Each operation is represented as its own module so the placement policy
        can map it to a specific node (edge/cloud).
        """
        app_name = "JSSP_Shop"
        app = Application(name=app_name)

        for op in self.operation_specs:
            job_id = op["job_id"]
            op_idx = op["op_idx"]
            duration = op["duration"]

            source_name = f"Source_Job_{job_id}"
            start_time = job_release_times.get(job_id, 0)

            current_module = f"Op_{job_id}_{op_idx}"

            if op_idx == 0:
                # Source and first hop
                msg_start_name = f"Msg_J{job_id}_Start"
                msg_start = Message(
                    msg_start_name,
                    source_name,
                    current_module,
                    instructions=duration,
                    bytes=100,
                )
                msg_start.last_idDes = []
                app.messages[msg_start_name] = msg_start
                dist = deterministic_distribution(name=f"Start_Dist_J{job_id}", time=start_time)
                app.add_service_source(source_name, dist, msg_start)
            else:
                prev_module = f"Op_{job_id}_{op_idx-1}"
                prev_idx = op_idx - 1
                if prev_idx == 0:
                    msg_in_name = f"Msg_J{job_id}_Start"
                else:
                    msg_in_name = f"Msg_J{job_id}_Op{prev_idx-1}_to_Op{prev_idx}"
                msg_in = app.messages[msg_in_name]

                msg_out_name = f"Msg_J{job_id}_Op{op_idx-1}_to_Op{op_idx}"
                msg_out = Message(
                    msg_out_name,
                    prev_module,
                    current_module,
                    instructions=duration,
                    bytes=100,
                )
                msg_out.last_idDes = []
                app.messages[msg_out_name] = msg_out

                # Register transition on the previous module
                app.add_service_module(
                    prev_module,
                    message_in=msg_in,
                    message_out=msg_out,
                    distribution=lambda **_: True,
                    module_dest=[current_module],
                    p=[1.0],
                )

            # Last operation: add sink rule
            if op_idx == len(self.raw_jobs[job_id]) - 1:
                msg_in_name = f"Msg_J{job_id}_Op{op_idx-1}_to_Op{op_idx}" if op_idx > 0 else f"Msg_J{job_id}_Start"
                msg_in = app.messages[msg_in_name]
                app.add_service_module(
                    current_module,
                    message_in=msg_in,
                    message_out=None,
                    distribution=lambda **_: True,
                    module_dest=[],
                    p=[],
                )

        # Apply the placement mapping so YAFS knows which nodes host each operation
        app.module_node_map = module_node_map
        return app
