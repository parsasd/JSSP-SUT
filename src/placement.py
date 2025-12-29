from yafs.placement import Placement
from yafs.application import Application

class JSSPPlacement(Placement):
    """
    Maps:
    - 'Source_Job_X' -> Node 0 (Cloud/Controller)
    - 'Machine_X'    -> Node X+1 (Edge Nodes 1..15)
    """
    def initial_allocation(self, sim, app_name):
        allocations = []
        app = sim.apps[app_name]
        
        # In YAFS, app.services contains the list of services per module.
        # We iterate module names.
        for module_name in app.services.keys():
            
            # 1. Check if it's a Source
            if module_name.startswith("Source"):
                # Deploy sources on Node 0
                allocations.append((module_name, 0))
            
            # 2. Check if it's a Machine
            elif module_name.startswith("Machine_"):
                # Extract ID: "Machine_5" -> 5
                try:
                    m_id = int(module_name.split("_")[1])
                    # Taillard Machines 0-14 map to Topology Nodes 1-15
                    node_id = m_id + 1
                    sim.deploy_module(app_name, module_name, app.services[module_name], ids=[node_id])
                except ValueError:
                    print(f"Warning: Could not parse machine ID from {module_name}")
        
        return allocations