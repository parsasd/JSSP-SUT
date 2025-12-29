import copy
import random

import yafs.core
from yafs.application import Application


def patch_deploy_module():
    """
    YAFS' deploy_module drops the probability list (`p`) from services.
    That causes KeyError in __add_consumer_module. We reintroduce it here.
    """
    if getattr(yafs.core.Sim, "_deploy_patched", False):
        return

    def deploy_module(self, app_name, module, services, ids):
        register_consumer_msg = []
        id_DES = []

        for service in services:
            if service["type"] == Application.TYPE_SOURCE:
                # Generate pure sources as in the original method
                for id_topology in ids:
                    id_DES.append(
                        self._Sim__deploy_source_module(
                            app_name,
                            module,
                            distribution=service["dist"],
                            msg=service["message_out"],
                            id_node=id_topology,
                        )
                    )
            else:
                register_consumer_msg.append(
                    {
                        "message_in": service["message_in"],
                        "message_out": service["message_out"],
                        "module_dest": service["module_dest"],
                        "dist": service["dist"],
                        "param": service["param"],
                        "p": service.get("p", []),
                    }
                )

        if register_consumer_msg:
            for id_topology in ids:
                id_DES.append(
                    self._Sim__deploy_module(
                        app_name, module, id_topology, register_consumer_msg
                    )
                )

        return id_DES

    yafs.core.Sim.deploy_module = deploy_module
    yafs.core.Sim._deploy_patched = True


def patch_consumer_module():
    """
    Make __add_consumer_module resilient when last_idDes is missing.
    """
    if getattr(yafs.core.Sim, "_consumer_patched", False):
        return

    def _patched_add_consumer_module(self, ides, app_name, module, register_consumer_msg):
        self.logger.debug("Added_Process - Module Consumer: %s\t#DES:%i" % (module, ides))
        while not self.stop and self.des_process_running[ides]:
            if self.des_process_running[ides]:
                msg = yield self.consumer_pipes["%s%s%i" % (app_name, module, ides)].get()

                # Ensure the inbound message has a history list
                if not hasattr(msg, "last_idDes") or msg.last_idDes is None:
                    msg.last_idDes = []

                doBefore = False
                for register in register_consumer_msg:
                    # --- CRITICAL FIX START ---
                    # Skip services that don't have a valid input message (e.g. placeholders)
                    if not register["message_in"]:
                        continue
                    # --- CRITICAL FIX END ---

                    if msg.name != register["message_in"].name:
                        continue

                    if not doBefore:
                        self.logger.debug(
                            "(App:%s#DES:%i#%s)\tModule - Recording the message:\t%s"
                            % (app_name, ides, module, msg.name)
                        )
                        type = self.NODE_METRIC

                        service_time = self._Sim__update_node_metrics(app_name, module, msg, ides, type)
                        yield self.env.timeout(service_time)
                        doBefore = True

                    if not register["message_out"]:
                        self.logger.debug(
                            "(App:%s#DES:%i#%s)\tModule - Sink Message:\t%s"
                            % (app_name, ides, module, msg.name)
                        )
                        continue

                    if register["dist"](**register["param"]):
                        msg_out = copy.copy(register["message_out"])
                        msg_out.timestamp = self.env.now
                        msg_out.id = msg.id

                        last_ids = list(msg.last_idDes) if msg.last_idDes is not None else []
                        last_ids.append(ides)
                        msg_out.last_idDes = last_ids

                        if not register["module_dest"]:
                            self._Sim__send_message(app_name, msg_out, ides, self.FORWARD_METRIC)
                        else:
                            for idx, module_dst in enumerate(register["module_dest"]):
                                probs = register.get("p", [])
                                prob_val = probs[idx] if idx < len(probs) else 1.0
                                if random.random() <= prob_val:
                                    self._Sim__send_message(app_name, msg_out, ides, self.FORWARD_METRIC)
                    else:
                        self.logger.debug(
                            "(App:%s#DES:%i#%s)\tModule - Stopped Message:\t%s"
                            % (app_name, ides, module, register["message_out"].name)
                        )

        self.logger.debug("STOP_Process - Module Consumer: %s\t#DES:%i" % (module, ides))

    yafs.core.Sim._Sim__add_consumer_module = _patched_add_consumer_module
    yafs.core.Sim._consumer_patched = True


def patch_source_single_shot():
    """
    Allow single-shot sources by honoring a distribution attr `single_shot`.
    """
    if getattr(yafs.core.Sim, "_source_single_shot_patched", False):
        return

    def _patched_add_source_population(self, idDES, name_app, message, distribution):
        self.logger.debug("Added_Process - Module Pure Source\t#DES:%i" % idDES)
        first = True
        while not self.stop and self.des_process_running[idDES]:
            nextTime = distribution.next()
            yield self.env.timeout(nextTime)
            if self.des_process_running[idDES]:
                self.logger.debug("(App:%s#DES:%i)\tModule - Generating Message: %s \t(T:%d)" % (name_app, idDES, message.name,self.env.now))

                msg = copy.copy(message)
                msg.timestamp = self.env.now
                msg.id = self._Sim__getIDMessage()
                msg.original_DES_src = idDES
                self._Sim__send_message(name_app, msg, idDES, self.SOURCE_METRIC)

                if getattr(distribution, "single_shot", False):
                    self.des_process_running[idDES] = False
                    break

            first = False

        self.logger.debug("STOP_Process - Module Pure Source\t#DES:%i" % idDES)

    yafs.core.Sim._Sim__add_source_population = _patched_add_source_population
    yafs.core.Sim._source_single_shot_patched = True


def apply_yafs_patches():
    patch_deploy_module()
    patch_consumer_module()
    patch_source_single_shot()
    # Silence verbose placement table printed at run start
    if not getattr(yafs.core.Sim, "_silenced_print_assignments", False):
        yafs.core.Sim.print_debug_assignaments = lambda self: None
        yafs.core.Sim._silenced_print_assignments = True