import csv
from typing import Any, Dict, Optional


class Metrics:
    """
    Helper class to record simulation events and link metrics in CSV files.

    Two CSV files are created:

    - ``<path>.csv``      : event-level metrics (per message / DES event)
    - ``<path>_link.csv`` : link-level metrics (per hop across the topology)
    """

    # Time-related metric identifiers
    TIME_LATENCY = "time_latency"
    TIME_WAIT = "time_wait"
    TIME_RESPONSE = "time_response"
    TIME_SERVICE = "time_service"
    TIME_TOTAL_RESPONSE = "time_total_response"

    # Energy-related metric identifiers
    WATT_SERVICE = "byService"
    WATT_UPTIME = "byUptime"

    def __init__(self, default_results_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        default_results_path:
            Base path (without extension) for the CSV output. If ``None``,
            the default value ``"result"`` is used, generating
            ``result.csv`` and ``result_link.csv``.
        """
        columns_event = [
            "id",
            "type",
            "app",
            "module",
            "message",
            "DES.src",
            "DES.dst",
            "TOPO.src",
            "TOPO.dst",
            "module.src",
            "service",
            "time_in",
            "time_out",
            "time_emit",
            "time_reception",
        ]
        columns_link = [
            "id",
            "type",
            "src",
            "dst",
            "app",
            "latency",
            "message",
            "ctime",
            "size",
            "buffer",
        ]

        path = "result"
        if default_results_path is not None:
            path = default_results_path

        # Event-level metrics file
        self.__filef = open(f"{path}.csv", "w", newline="")
        # Link-level metrics file
        self.__filel = open(f"{path}_link.csv", "w", newline="")

        self.__ff = csv.writer(self.__filef)
        self.__ff_link = csv.writer(self.__filel)

        # Write headers
        self.__ff.writerow(columns_event)
        self.__ff_link.writerow(columns_link)

    def flush(self) -> None:
        """
        Flush buffered data to the CSV files.
        """
        self.__filef.flush()
        self.__filel.flush()

    def insert(self, value: Dict[str, Any]) -> None:
        """
        Insert a new event-level metric row.

        Parameters
        ----------
        value:
            Dictionary containing all fields required by the event header
            (e.g. ``id``, ``type``, ``app``, ``module``, ...).
        """
        self.__ff.writerow(
            [
                value["id"],
                value["type"],
                value["app"],
                value["module"],
                value["message"],
                value["DES.src"],
                value["DES.dst"],
                value["TOPO.src"],
                value["TOPO.dst"],
                value["module.src"],
                value["service"],
                value["time_in"],
                value["time_out"],
                value["time_emit"],
                value["time_reception"],
            ]
        )

    def insert_link(self, value: Dict[str, Any]) -> None:
        """
        Insert a new link-level metric row.

        Parameters
        ----------
        value:
            Dictionary containing all fields required by the link header
            (e.g. ``id``, ``type``, ``src``, ``dst``, ...).
        """
        self.__ff_link.writerow(
            [
                value["id"],
                value["type"],
                value["src"],
                value["dst"],
                value["app"],
                value["latency"],
                value["message"],
                value["ctime"],
                value["size"],
                value["buffer"],
            ]
        )

    def close(self) -> None:
        """
        Close the underlying CSV files.

        After calling this method, no further writes should be attempted.
        """
        self.__filef.close()
        self.__filel.close()
