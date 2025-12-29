from yafs.placement import Placement
from yafs.application import Application


class JSSPPlacement(Placement):
    """
    Placement policy that maps sources to node 0 and operation modules to nodes
    chosen by the optimizer.
    """

    def __init__(self, module_node_map, name="JSSPPlacement", logger=None):
        super().__init__(name=name, logger=logger)
        self.module_node_map = module_node_map

    def initial_allocation(self, sim, app_name):
        allocations = []
        app = sim.apps[app_name]

        for module_name, services in app.services.items():
            # Sources live on node 0
            if module_name.startswith("Source"):
                allocations.append((module_name, 0))
                continue

            # All operation modules are mapped by the GA
            target_node = self.module_node_map.get(module_name)
            if target_node is None:
                continue
            sim.deploy_module(app_name, module_name, services, ids=[target_node])

        return allocations
