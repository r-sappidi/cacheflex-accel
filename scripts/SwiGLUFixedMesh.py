"""Preserve the pilot's 2x4 fabric and LLC/directory placement at one CPU."""
from topologies.Mesh_XY import Mesh_XY


class SwiGLUFixedMesh(Mesh_XY):
    description = 'Pilot eight-router mesh with missing private controllers'

    def makeTopology(self, options, network, IntLink, ExtLink, Router):
        assert options.num_cpus in (1, 8)
        assert options.num_l2caches == options.num_dirs == 8
        routers=[Router(router_id=i,latency=options.router_latency) for i in range(8)]
        network.routers=routers
        external=[]; internal=[]; link_id=0
        counts = {}
        for node in self.nodes:
            typ = node.type
            assert 'DMA' not in typ, 'SE pilot has no DMA controllers'
            index=counts.get(typ,0);counts[typ]=index+1
            assert index<8
            external.append(ExtLink(link_id=link_id,ext_node=node,
                int_node=routers[index],latency=options.link_latency))
            link_id+=1
        network.ext_links=external
        # Same directional wiring/weights as Mesh_XY, in the same order.
        for direction in ('East','West','North','South'):
            edges=[]
            if direction in ('East','West'):
                for row in range(2):
                    for col in range(3):
                        a=row*4+col;b=a+1
                        edges.append((a,b) if direction=='East' else (b,a))
            else:
                edges=[(col,col+4) if direction=='North' else (col+4,col) for col in range(4)]
            opposite=dict(East='West',West='East',North='South',South='North')[direction]
            for src,dst in edges:
                internal.append(IntLink(link_id=link_id,src_node=routers[src],dst_node=routers[dst],
                    src_outport=direction,dst_inport=opposite,latency=options.link_latency,
                    weight=1 if direction in ('East','West') else 2))
                link_id+=1
        network.int_links=internal
