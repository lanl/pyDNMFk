# @author: Ismael Boureima
import argparse
import os, sys,time
from .toolz import log, amber, blue, green, purple, red
import cupy as cp
try:
    from cupy.cuda import nccl
    NCCL_backend = 1
except:
    NCCL_backend = 0

try:
    import asyncio
    import ucp
    UCX_backend = 1
except:
    UCXL_backend = 0

import numpy as np
from pyDNMFk.communicators import NCCLComm
from mpi4py import MPI
from pyDNMFk.get_reduce_comm_tree import getReduceCommTree


class GetTopology():
    r"""
    Constructor for the GetTopology class.

    This class is used for identifying the system topology in the context of MPI.
    It determines the global and local ranks, identifies the host system, and
    determines GPU availability and assignments.

    Args:
    - comm (MPI Comm): The MPI communicator. Defaults to MPI.COMM_WORLD.
    - gmasterRank (int): Rank of the global master. Defaults to 0.
    """
    def __init__(self, comm=None, gmasterRank=0):
        if comm is None: comm = MPI.COMM_WORLD
        rank        = comm.Get_rank()
        size        = comm.Get_size()
        myHost      = os.uname()[1]
        GMASTER     = (rank==gmasterRank)
        nGPUs_local = cp.cuda.runtime.getDeviceCount()
        self.comm, self.rank, self.size, self.myHost, self.GMASTER, self.nGPU_local = comm, rank, size, myHost, GMASTER, nGPUs_local

        if GMASTER:
            hosts, local_size, members = [myHost], {myHost:1}, {myHost:[gmasterRank]}
            for r in range(size):
              if r != gmasterRank:
                rh = comm.recv(source=r, tag=10*r)
                if rh not in hosts:
                    hosts.append(rh)
                    local_size[rh]=0
                    members[rh] = []
                local_size[rh] += 1
                if r not in members[rh]: members[rh].append(r)
        else:
            hosts, local_size = None, None
            comm.send(myHost, dest=gmasterRank, tag=10*rank)
        comm.barrier()

        if GMASTER:
            #if rank == 0: print("[+] OK020")
            global_uID = None
            if NCCL_backend:
              #if rank == 0: print("[+] OK021")
              try:
                global_uID = nccl.get_unique_id()
                #if rank == 0: print("[+] OK022")
              except:
                if rank == 0: print("[!] UNABLE to get NCCL GLOBAL_UID")
            nHosts  = len(hosts)
            stride  = int(size/nHosts)
            MPI_TOPO = {'GMASTER':rank, 'hosts':hosts, 'nHosts':nHosts, 'stride':stride, 'nccl_GUID':global_uID, 'local_size':local_size, 'members':members}
            MPI_TOPO['LMASTER'] = {}
            #if rank == 0: print("[+] OK023")
            for host in hosts:
                idx     = hosts.index(host)
                MPI_TOPO['LMASTER'][host]=idx
        else:
            MPI_TOPO = None
        comm.barrier()

        MPI_TOPO   = comm.bcast(MPI_TOPO, root=0)
        #if rank == 0: print("[+] OK04")
        nHosts     = MPI_TOPO['nHosts']
        stride     = MPI_TOPO['stride']
        hosts      = MPI_TOPO['hosts']
        local_size = MPI_TOPO['local_size']
        idx        = hosts.index(myHost)
        lrank      = rank // nHosts
        myMaster   = MPI_TOPO['LMASTER'][myHost]
        LMASTER    = (lrank==0)
        self.topology, self.nHost, self.hosts, self.host_idx, self.mpi_stride = MPI_TOPO, nHosts, hosts, idx, stride
        self.local_size, self.lrank, self.myMaster, self.LMASTER              = local_size, lrank, myMaster, LMASTER

        if GMASTER:
            gIDs      = {myHost:np.arange(nGPUs_local, dtype=np.int32)}
            nGPUs     = nGPUs_local
            lastID    = gIDs[myHost][-1]
            for r in list(MPI_TOPO['LMASTER'].keys()):
                if MPI_TOPO['LMASTER'][r] != rank:
                    ng      = comm.recv(source=MPI_TOPO['LMASTER'][r], tag=100*MPI_TOPO['LMASTER'][r])
                    nGPUs   += ng
                    gIDs[r]  = np.arange(ng, dtype=np.int32)+lastID+1
                    lastID  = gIDs[r][-1]
            nGPUs = min(nGPUs, size)
            gIDs['nGPUs'] = min(nGPUs, size)
        elif ( LMASTER and (not GMASTER) ):
            gIDs   = None
            ng     = None
            comm.send(nGPUs_local, dest=MPI_TOPO['GMASTER'], tag=100*rank)
        else:
            ng     = None
            gIDs   = None
        comm.barrier()
        #if rank == 0: print("[+] OK06")
        gIDs   = comm.bcast(gIDs, root=MPI_TOPO['GMASTER'])
        #if rank == 0: print("[+] OK07")
        nGPUs  = gIDs['nGPUs']
        del gIDs['nGPUs']
        MPI_TOPO['gIDs']  = gIDs
        MPI_TOPO['nGPUs'] = nGPUs
        if lrank < nGPUs_local:
            self.gID = MPI_TOPO['gIDs'][myHost][lrank]

    def showTopology(self):
        """
        Method to display the detected topology for debugging purposes.
        """
        if  self.GMASTER:
             role = f"<{red('GMASTER')}>"
        elif  self.LMASTER and (not self.GMASTER):      
             role = f"<{blue('LMASTER')}>"
        else:
             role = f"<{amber('WORKER')}>"
        log(msg = f"{role} on host {self.myHost}", rank=self.rank, lrank=self.lrank)
        self.comm.barrier()

    def getReduceCommTree(self, VRBZ=False):
        """
        Retrieve a communication tree for efficient reduce operations.

        Args:
        - VRBZ (bool): Verbosity level for debug outputs. Defaults to False.
        """
        x = np.arange(self.topology['nGPUs'])
        self.reduceCommTree = getReduceCommTree(x=x, VRBZ=VRBZ)
        
    


class superCommunicator():
    r"""
    Constructor for the superCommunicator class.

    This class acts as a wrapper around the MPI and NCCL communication protocols,
    enabling simplified and efficient communication setup based on the detected
    system topology.

    Args:
    - comm (MPI Comm): The MPI communicator.
    - gmasterRank (int): Rank of the global master. Defaults to 0.
    """
    def __init__(self, comm, gmasterRank=0):
        self.globalComm = comm
        self.topology   = GetTopology(comm=comm, gmasterRank=gmasterRank)
        self.comm       = {'GLOBAL':comm}
        self.comms      = ['GLOBAL']
        self.rank       = self.topology.rank
        self.GMASTER    = self.topology.GMASTER
        self.LMASTER    = self.topology.LMASTER
        self.myMaster   = self.topology.myMaster
        self.inContext  = {'GLOBAL':True}
        self.contexts  = list(self.inContext.keys())

    def createComm(self, root, members,name, backEnd='NCCL'):
        """
        Create a new communication context based on the provided parameters.

        Args:
        - root (int): Root rank for this communicator.
        - members (list): List of member ranks included in this communicator.
        - name (str): Name for this communicator.
        - backEnd (str): Communication backend (e.g., 'NCCL'). Defaults to 'NCCL'.
        """
        if name in self.comms:
            if self.GMASTER: 
                log(msg = f"[{red('!')} ]Communcator {name} existAlready", rank=self.rank, lrank=self.lrank)
            return
        self.comm[name] = NCCL_COMM(comm=self.globalComm, root=root, members=members, name=name)
        if self.rank in members:
            self.inContext[name]=True
        else:
            self.inContext[name]=False
        self.contexts  = list(self.inContext.keys())


class NCCL_COMM():
    r"""
    Constructor for the NCCL_COMM class.

    This class encapsulates NCCL-specific communication functions, enabling
    efficient GPU-to-GPU communications.

    Args:
    - comm (MPI Comm): The MPI communicator.
    - root (int): Root rank for this communicator.
    - members (list): List of member ranks included in this communicator.
    - name (str): Name for this communicator.
    """
    def __init__(self, comm, root, members, name):
        if comm.rank in members:
            self.rank  = comm.rank
            self.lrank = members.index(comm.rank)
            self.size  = len(members) 
            self.ROOT  = root
            uID = None
            self.MASTER = (comm.rank==root)
            if self.MASTER:
                uID = nccl.get_unique_id()
                print(f"[+] <LMASTER>[{comm.rank}]: uID = {uID[:8]}")
                for m in members:
                    if m != root: comm.send(uID, dest=m, tag=1000*m)
            else:
                uID = comm.recv(source=root, tag=1000*self.rank)
                print(f"[+] <WORKER>[{comm.rank}]: uID = {uID[:8]}")
            #comm.barrier()
            if uID is not None:
               print(f"   ->  Building NCCL comm [{len(members)}, uID:{uID[:8]}, lrank:{self.lrank}]")
               comm.barrier()
               ncclcomm = nccl.NcclCommunicator.initAll(members)
            if self.MASTER: log(msg = f"[{green('+')}] NCCL comm {name} built OK", rank=self.rank, lrank=self.lrank)
        else:
             ncclcomm = None
        self.comm = ncclcomm
