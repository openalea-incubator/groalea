# Author: C. Pradal
# Licence : GPL
from openalea.groimp.client import *
import openalea.groimp.graphio as io
from openalea.plantgl.all import Viewer

xlcode = """
import de.grogra.imp3d.objects.*;
public class Model {
    public void run() {
        String t = "Foo " + super.toString();
    }
}
"""
xml_graph = """
<graph xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <root root_id="1"/>

  <type name="Boid">
    <extends name="sphere"/>
  </type>

  <node id="1" name="C1" type="cylinder">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="2" name="C2" type="cylinder">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="3" name="C3" type="cylinder">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="4" name="S4" type="Boid">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="7" name="T7" type="node">
    	<property name="transformation">
            <matrix>
			1 0 0 0
			0 1 0 0
			0 0 1 1
			0 0 0 1    	
    	    </matrix>
        </property>
  </node>
  <node id="5" name="C5" type="cylinder">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="6" name="C6" type="cylinder">
    <property name="height" value="1"/>
    <property name="radius" value="0.2" />
  </node>
  <node id="8" name="T8" type="node">
    <property name="transformation">
    	<matrix>
			1 0 0 0
			0 0.866 0.5 0
			0 -0.5 0.866 0
			0 0 0 1    	
    	</matrix>
    </property>
  </node>
  <node id="9" name="T9" type="node">
    <property name="transformation">
    	<matrix>
			1 0 0 0
			0 0.866 -0.5 0
			0 0.5 0.866 0
			0 0 0 1    	
    	</matrix>
    </property>
  </node>
  <edge id="1" src_id="1" dest_id="2" type="successor" />
  <edge id="2" src_id="2" dest_id="3" type="successor" />
  <edge id="3" src_id="3" dest_id="7" type="successor" />
  <edge id="6" src_id="7" dest_id="4" type="successor" />
  <edge id="4" src_id="2" dest_id="8" type="branch" />
  <edge id="5" src_id="2" dest_id="9" type="branch" />
  <edge id="7" src_id="8" dest_id="5" type="successor" />
  <edge id="8" src_id="9" dest_id="6" type="successor" />
</graph>
"""
conn = connexion('localhost', '4711')
out_graph = simulation(xlcode, xml_graph, '', conn)

g, s = io.xml2graph(out_graph)
Viewer.display(s)

gxml = io.graph2xml(g)
