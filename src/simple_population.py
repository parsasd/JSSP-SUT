from yafs.population import Population
from yafs.application import Application


class SimplePopulation(Population):
    """Deploys every source service of an app to a single topology node."""

    def __init__(self, node_id=0, name="SimplePopulation", logger=None, single_shot=True):
        super(SimplePopulation, self).__init__(name=name, logger=logger)
        self.node_id = node_id
        self.single_shot = single_shot

    def initial_allocation(self, sim, app_name):
        """
        YAFS calls this before the simulation starts. We iterate over the
        registered services and deploy each SOURCE on the chosen node.
        """
        app = sim.apps[app_name]

        # app.services holds lists of service definitions per module
        for svc_list in app.services.values():
            for svc in svc_list:
                if svc.get("type") != Application.TYPE_SOURCE:
                    continue
                msg = svc.get("message_out")
                dist = svc.get("dist")
                if msg is None or dist is None:
                    # Nothing to deploy without a message or distribution
                    continue
                # Mark distribution for single emission if desired
                if self.single_shot:
                    setattr(dist, "single_shot", True)
                sim.deploy_source(app_name, id_node=self.node_id, msg=msg, distribution=dist)
