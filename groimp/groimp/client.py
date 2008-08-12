# licence

import httplib, urllib


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

    def send( self, xl_code, xml_graph = None ):
        conn = httplib.HTTPConnection(self.host, self.port)

        params, headers = self.url_encode(xl_code, xml_graph)
        conn.request("POST", "/cgi-bin/query", params, headers)

        response = conn.getresponse()
        data = response.read()
        conn.close()

        return data

    def url_encode(self, xl_code, xml_graph):
        """
        Encode the data into an http request.
        Returns the params and the headers, which are dict.
        """
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain"}
        params = {}
        params['xl'] = xl_code
        params['graph'] = xml_graph
        
        return params, headers
