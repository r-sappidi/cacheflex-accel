"""Fixed eight-router fabric for 1/4/8 CPU strong scaling.

LLC bank i and private hierarchy i attach to router i. Two memory directories
attach at routers0/4. Unlike Mesh_XY, router count does not shrink with CPUs.
"""
from topologies.BaseTopology import SimpleTopology


class MicroHeadMesh(SimpleTopology):
    description = 'Fixed 2x4 mesh, eight LLC banks, two memory directories'

    def __init__(self, controllers):
        self.nodes = controllers

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        assert options.num_cpus in (1,4,8)
        assert options.num_l2caches == 8 and options.num_dirs == 2
        routers=[Router(router_id=i,latency=options.router_latency) for i in range(8)]
        network.routers=routers
        ext=[]; internal=[]; link_id=0
        counts={}
        for node in self.nodes:
            typ=node.type; index=counts.get(typ,0); counts[typ]=index+1
            if 'Directory' in typ: router=index*4
            elif 'DMA' in typ: router=0
            else: router=index
            assert 0<=router<8,(typ,index)
            ext.append(ExtLink(link_id=link_id,ext_node=node,int_node=routers[router],latency=options.link_latency))
            link_id+=1
        network.ext_links=ext
        for row in range(2):
            for col in range(4):
                a=row*4+col
                if col<3:
                    for src,dst,outport,inport in [(a,a+1,'East','West'),(a+1,a,'West','East')]:
                        internal.append(IntLink(link_id=link_id,src_node=routers[src],dst_node=routers[dst],
                            src_outport=outport,dst_inport=inport,latency=options.link_latency,weight=1));link_id+=1
                if row==0:
                    for src,dst,outport,inport in [(a,a+4,'North','South'),(a+4,a,'South','North')]:
                        internal.append(IntLink(link_id=link_id,src_node=routers[src],dst_node=routers[dst],
                            src_outport=outport,dst_inport=inport,latency=options.link_latency,weight=2));link_id+=1
        network.int_links=internal

    def registerTopology(self, options):
        from common import FileSystemConfig
        from m5.params import MemorySize
        FileSystemConfig.register_node(list(range(options.num_cpus)),MemorySize(options.mem_size),0)
