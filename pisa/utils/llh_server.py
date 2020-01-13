#!/usr/bin/env python

"""
Server(s) for handling llh requests from a client: client passes free param
values, server sets these on its DistributionMaker, generates outputs, and
compares the resulting distributions against a reference template, returning
the llh value.

Code adapted from Dan Krause
  https://gist.github.com/dankrause/9607475
see `__license__`.
"""

from __future__ import absolute_import, division, print_function

__author__ = "Dan Krause, adapted by J.L. Lanfranchi"

__license__ = """
Copyright 2017 Dan Krause

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = [
    "DFLT_HOST",
    "DFLT_PORT",
    "DFLT_NUM_SERVERS",
    "send_obj",
    "receive_obj",
    "serve",
    "fork_servers",
    "main",
]


from argparse import ArgumentParser
from multiprocessing import cpu_count, Process
import pickle
import socketserver
import struct

from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import MapSet


DFLT_HOST = "localhost"
DFLT_PORT = "9000"
DFLT_NUM_SERVERS = cpu_count()


class ConnectionClosed(Exception):
    """Connection closed"""


def send_obj(obj, sock):
    """Send a Python object over a socket. Object is pickle-encoded as the
    payload and sent preceded by a 4-byte header which indicates the number of
    bytes of the payload.

    Parameters
    ----------
    sock : socket
    obj : pickle-able Python object
        Object to send

    """
    # Turn object into a string
    payload = pickle.dumps(obj)

    # Create a header that says how large the payload is
    header = struct.pack('!i', len(payload))

    # Send header
    sock.sendall(header)

    # Send payload
    sock.sendall(payload)


def receive_obj(sock):
    """Receive an object from a socket. Payload is a pickle-encoded object, and
    header (prefixing payload) is 4-byte int indicating length of the payload.

    Parameters
    ----------
    sock : socket

    Returns
    -------
    obj
        Unpickled Python object

    """
    # Get 4-byte header which tells how large the subsequent payload will be
    header = sock.recv(4)
    if len(header) == 0:
        raise ConnectionClosed()
    payload_size = struct.unpack('!i', header)[0]

    # Receive the payload
    payload = sock.recv(payload_size)
    if len(payload) == 0:
        raise ConnectionClosed()

    # Payload was pickled; unpickle to recreate original Python object
    obj = pickle.loads(payload)

    return obj


def serve(config, ref, port=DFLT_PORT):
    """Instantiate PISA objects and run server for processing requests.

    Parameters
    ----------
    config : str or iterable thereof
        Resource path(s) to pipeline config(s)

    ref : str
        Resource path to reference map

    port : int or str, optional

    """
    # Instantiate the objects here to save having to do this repeatedly
    dist_maker = DistributionMaker(config)
    ref = MapSet.from_json(ref)

    # Define server as a closure such that it captures the above-instantiated objects
    class MyTCPHandler(socketserver.BaseRequestHandler):
        """
        The request handler class for our server.

        It is instantiated once per connection to the server, and must override
        the handle() method to implement communication to the client.

        See socketserver.BaseRequestHandler for documentation of args.
        """
        def handle(self):
            try:
                param_values = receive_obj(self.request)
            except ConnectionClosed:
                return
            dist_maker._set_rescaled_free_params(param_values)  # pylint: disable=protected-access
            test_map = dist_maker.get_outputs(return_sum=True)[0]
            llh = test_map.llh(
                expected_values=ref,
                binned=False,  # return sum over llh from all bins (not per-bin llh's)
            )
            send_obj(llh, self.request)

    server = socketserver.TCPServer((DFLT_HOST, int(port)), MyTCPHandler)
    print("llh server started on {}:{}".format(DFLT_HOST, port))
    server.serve_forever()


def fork_servers(config, ref, port=DFLT_PORT, num=DFLT_NUM_SERVERS):
    """Fork multiple servers for handling LLH requests. Objects are identically
    configured, and ports used are sequential starting from `port`.

    Parameters
    ----------
    config : str or iterable thereof
    ref : str
    port : str or int, optional
    num : int, optional
        Defaults to number of CPUs returned by `multiple.cpu_count()`

    """
    processes = []
    for port_ in range(int(port), int(port) + int(num)):
        kwargs = dict(config=config, ref=ref, port=str(port_))
        process = Process(target=serve, kwargs=kwargs)
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


def main(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        nargs="+",
        help="""Resource location of one or more pipeline configs""",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Resource location of reference (truth) map",
    )
    parser.add_argument("--port", default=DFLT_PORT)
    parser.add_argument(
        "--num",
        default=1,
        type=int,
        help="Number of servers to fork (>= 1); if set to 1, no forking occurs",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    num = kwargs.pop("num")
    if num == 1:
        serve(**kwargs)
    else:
        fork_servers(num=num, **kwargs)


if __name__ == "__main__":
    main()
