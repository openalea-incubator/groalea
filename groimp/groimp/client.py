# licence

import httplib, urllib
import re

class GroIMPClient(object):
    """
    GroIMPClient communicates with GroIMP through the network:
      - it creates a socket,
      - it connects to the server,
      - creates a new project in GroIMP,
      - sends the XL file, 
      - receive an xml file containing Graph and geometry, 
      - export this file into PlantGL scene graph.
    """
    def __init__( self, host='localhost', port = '58090'):
        """
        `host` is the remote address of the server.
        `port` is the same port used by the server.
        """
        self.host = host
        self.port = port

    def send( self, xl_code, xml_graph = None, command = '' ):
        conn = httplib.HTTPConnection(self.host, self.port)

        params, headers = self.url_encode(xl_code, xml_graph, command)
        conn.request("POST", "/test.html", params, headers)

        response = conn.getresponse()
        data = response.read()
        conn.close()
        
        # We receive the whole web page.
        # Just extract the graph only...
        start = re.search('<graph', data).start()
        end = re.search('</graph>', data).end()
        
        return data[start:end]

    def url_encode(self, xl_code, xml_graph, command):
        """
        Encode the data into an http request.
        Returns the params and the headers, which are dict.
        """
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain"}
        params = {}
        params['xlcode'] = xl_code
        params['graph'] = xml_graph
        if command:
            params['command'] = command
        p = urllib.urlencode(params)
        h = headers
        return p, h


def simulation(xl_code, graph, command, connexion=None):
    """
    Send to the GroIMP server the xl_code and the graph for the simulation.
    Returns the resulting graph.
    """
    if not connexion:
        connexion = GroIMPClient()
    return connexion.send(xl_code, graph, command)

def connexion(host, port):
    return GroIMPClient(host, port)

if __name__ == '__main__':
    conn = connexion('localhost', '4711')
    xlcode = """
import de.grogra.imp3d.objects.*;
public class MyBox extends Box {
    public String toString() {
        return "Foo " + super.toString();
    }
}
"""
    xml_graph = """
<graph>
  <!--Zero or more repetitions:-->
  <type name="string">
    <extends name="string"/>
    <implements name="string"/>
  </type>
  <!--Zero or more repetitions:-->
  <node id="3" type="sphere" name="s1">
    <property name="radius" value="1.0"/>
    <property name="transform">
    	<matrix>
    		1 0 0 2
    		0 1 0 3
    		0 0 1 4
    		0 0 0 1
    	</matrix>
    </property>
  </node>
  <root root_id="3"/>
  <!--Zero or more repetitions:-->
  <edge id="3" src_id="3" dest_id="3" type="successor"/>
</graph>
"""
    simulation(xlcode, xml_graph, '', conn)
