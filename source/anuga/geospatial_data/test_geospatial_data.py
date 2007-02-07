#!/usr/bin/env python


import unittest
import os
from Numeric import zeros, array, allclose, concatenate
from math import sqrt, pi
import tempfile
from sets import ImmutableSet

from anuga.geospatial_data.geospatial_data import *
from anuga.coordinate_transforms.geo_reference import Geo_reference, TitleError
from anuga.coordinate_transforms.redfearn import degminsec2decimal_degrees
from anuga.utilities.anuga_exceptions import ANUGAError
# Ignore these warnings, since we still want to test .xya code.
import warnings
warnings.filterwarnings(action = 'ignore',
                        message='.xya format is deprecated.  Please use .txt.',
                        category=DeprecationWarning)

warnings.filterwarnings(action = 'ignore',
                        message='The text file values must be ab',
                        category=DeprecationWarning)

warnings.filterwarnings(action = 'ignore',
                        message='Text file format is moving to comma se',
                        category=DeprecationWarning)

warnings.filterwarnings(action = 'ignore',
                        message='Specifying delimiters will be removed.',
                        category=DeprecationWarning)

class Test_Geospatial_data(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_0(self):
        #Basic points
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[1.0, 2.1], [3.0, 5.3]]
        G = Geospatial_data(points)

        assert allclose(G.data_points, [[1.0, 2.1], [3.0, 5.3]])

        #Check __repr__

        rep = `G`
        ref = '[[ 1.   2.1]\n [ 3.   5.3]]'
        assert rep == ref


        #Check getter
        assert allclose(G.get_data_points(), [[1.0, 2.1], [3.0, 5.3]])
        
        #Check defaults
        assert G.attributes is None
        
        assert G.geo_reference.zone == Geo_reference().zone
        assert G.geo_reference.xllcorner == Geo_reference().xllcorner
        assert G.geo_reference.yllcorner == Geo_reference().yllcorner
        

    def test_1(self):
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = [2, 4]
        G = Geospatial_data(points, attributes)        

        assert G.attributes.keys()[0] == 'attribute'
        assert allclose(G.attributes.values()[0], [2, 4])
        

    def test_2(self):
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = [2, 4]
        G = Geospatial_data(points, attributes,
                            geo_reference=Geo_reference(56, 100, 200))

        assert G.geo_reference.zone == 56
        assert G.geo_reference.xllcorner == 100
        assert G.geo_reference.yllcorner == 200


    def test_get_attributes_1(self):
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = [2, 4]
        G = Geospatial_data(points, attributes,
                            geo_reference=Geo_reference(56, 100, 200))


        P = G.get_data_points(absolute=False)
        assert allclose(P, [[1.0, 2.1], [3.0, 5.3]])        

        P = G.get_data_points(absolute=True)
        assert allclose(P, [[101.0, 202.1], [103.0, 205.3]])        

        V = G.get_attributes() #Simply get them
        assert allclose(V, [2, 4])

        V = G.get_attributes('attribute') #Get by name
        assert allclose(V, [2, 4])

    def test_get_attributes_2(self):
        #Multiple attributes
        
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = {'a0': [0, 0], 'a1': [2, 4], 'a2': [79.4, -7]}
        G = Geospatial_data(points, attributes,
                            geo_reference=Geo_reference(56, 100, 200),
                            default_attribute_name='a1')


        P = G.get_data_points(absolute=False)
        assert allclose(P, [[1.0, 2.1], [3.0, 5.3]])        
        
        V = G.get_attributes() #Get default attribute
        assert allclose(V, [2, 4])

        V = G.get_attributes('a0') #Get by name
        assert allclose(V, [0, 0])

        V = G.get_attributes('a1') #Get by name
        assert allclose(V, [2, 4])

        V = G.get_attributes('a2') #Get by name
        assert allclose(V, [79.4, -7])

        try:
            V = G.get_attributes('hdnoatedu') #Invalid
        except AssertionError:
            pass
        else:
            raise 'Should have raised exception' 

    def test_get_data_points(self):
        points_ab = [[12.5,34.7],[-4.5,-60.0]]
        x_p = -10
        y_p = -40
        geo_ref = Geo_reference(56, x_p, y_p)
        points_rel = geo_ref.change_points_geo_ref(points_ab)
        
        spatial = Geospatial_data(points_rel, geo_reference=geo_ref)

        results = spatial.get_data_points(absolute=False)
        
        assert allclose(results, points_rel)
        
        x_p = -1770
        y_p = 4.01
        geo_ref = Geo_reference(56, x_p, y_p)
        points_rel = geo_ref.change_points_geo_ref(points_ab)
        results = spatial.get_data_points \
                  ( geo_reference=geo_ref)
        
        assert allclose(results, points_rel)

  
    def test_get_data_points_lat_long(self):
        # lat long [-30.],[130]
        #Zone:   52    
        #Easting:  596450.153  Northing: 6680793.777 
        # lat long [-32.],[131]
        #Zone:   52    
        #Easting:  688927.638  Northing: 6457816.509 
        
        points_Lat_long = [[-30.,130], [-32,131]]
        
        spatial = Geospatial_data(latitudes=[-30, -32.],
                                  longitudes=[130, 131])

        results = spatial.get_data_points(as_lat_long=True)
        #print "test_get_data_points_lat_long - results", results
        #print "points_Lat_long",points_Lat_long 
        assert allclose(results, points_Lat_long)
      
    def test_get_data_points_lat_longII(self):
        # x,y  North,east long,lat
        boundary_polygon = [[ 250000, 7630000]]
        zone = 50
        
        geo_reference = Geo_reference(zone=zone)
        geo = Geospatial_data(boundary_polygon,geo_reference=geo_reference)
        seg_lat_long = geo.get_data_points(as_lat_long=True)
        lat_result = degminsec2decimal_degrees(-21,24,54)
        long_result = degminsec2decimal_degrees(114,35,17.89)
        #print "seg_lat_long", seg_lat_long [0][0]
        #print "lat_result",lat_result 
        assert allclose(seg_lat_long[0][0], lat_result)#lat
        assert allclose(seg_lat_long[0][1], long_result)#long

              
    def test_set_geo_reference(self):
        points_ab = [[12.5,34.7],[-4.5,-60.0]]
        x_p = -10
        y_p = -40
        geo_ref = Geo_reference(56, x_p, y_p)
        points_rel = geo_ref.change_points_geo_ref(points_ab)
        spatial = Geospatial_data(points_rel)
        # since the geo_ref wasn't set
        assert not allclose( points_ab, spatial.get_data_points(absolute=True))
        
        spatial = Geospatial_data(points_rel, geo_reference=geo_ref)
        assert allclose( points_ab, spatial.get_data_points(absolute=True))
        
        x_p = 10
        y_p = 400
        new_geo_ref = Geo_reference(56, x_p, y_p)
        spatial.set_geo_reference(new_geo_ref)
        assert allclose( points_ab, spatial.get_data_points(absolute=True))
        
        
        
    def test_conversions_to_points_dict(self):
        #test conversions to points_dict
        
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = {'a0': [0, 0], 'a1': [2, 4], 'a2': [79.4, -7]}
        G = Geospatial_data(points, attributes,
                            geo_reference=Geo_reference(56, 100, 200),
                            default_attribute_name='a1')


        points_dict = geospatial_data2points_dictionary(G)

        assert points_dict.has_key('pointlist')
        assert points_dict.has_key('attributelist')        
        assert points_dict.has_key('geo_reference')

        assert allclose( points_dict['pointlist'], points )

        A = points_dict['attributelist']
        assert A.has_key('a0')
        assert A.has_key('a1')
        assert A.has_key('a2')        

        assert allclose( A['a0'], [0, 0] )
        assert allclose( A['a1'], [2, 4] )        
        assert allclose( A['a2'], [79.4, -7] )


        geo = points_dict['geo_reference']
        assert geo is G.geo_reference


    def test_conversions_from_points_dict(self):
        """test conversions from points_dict
        """

        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = {'a0': [0, 0], 'a1': [2, 4], 'a2': [79.4, -7]}

        points_dict = {}
        points_dict['pointlist'] = points
        points_dict['attributelist'] = attributes
        points_dict['geo_reference'] = Geo_reference(56, 100, 200)
        

        G = points_dictionary2geospatial_data(points_dict)

        P = G.get_data_points(absolute=False)
        assert allclose(P, [[1.0, 2.1], [3.0, 5.3]])        
        
        #V = G.get_attribute_values() #Get default attribute
        #assert allclose(V, [2, 4])

        V = G.get_attributes('a0') #Get by name
        assert allclose(V, [0, 0])

        V = G.get_attributes('a1') #Get by name
        assert allclose(V, [2, 4])

        V = G.get_attributes('a2') #Get by name
        assert allclose(V, [79.4, -7])

    def test_add(self):
        """ test the addition of two geospatical objects
            no geo_reference see next test
        """
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = {'depth':[2, 4], 'elevation':[6.1, 5]}
        attributes1 = {'depth':[2, 4], 'elevation':[2.5, 1]}
        G1 = Geospatial_data(points, attributes)        
        G2 = Geospatial_data(points, attributes1) 
        
#        g3 = geospatial_data2points_dictionary(G1)
#        print 'g3=', g3
        
        G = G1 + G2

        assert G.attributes.has_key('depth')
        assert G.attributes.has_key('elevation')
        assert allclose(G.attributes['depth'], [2, 4, 2, 4])
        assert allclose(G.attributes['elevation'], [6.1, 5, 2.5, 1])
        assert allclose(G.get_data_points(), [[1.0, 2.1], [3.0, 5.3],
                                              [1.0, 2.1], [3.0, 5.3]])
        
    def test_addII(self):
        """ test the addition of two geospatical objects
            no geo_reference see next test
        """
        points = [[1.0, 2.1], [3.0, 5.3]]
        attributes = {'depth':[2, 4]}
        G1 = Geospatial_data(points, attributes) 
        
        points = [[5.0, 2.1], [3.0, 50.3]]
        attributes = {'depth':[200, 400]}
        G2 = Geospatial_data(points, attributes)
        
#        g3 = geospatial_data2points_dictionary(G1)
#        print 'g3=', g3
        
        G = G1 + G2

        assert G.attributes.has_key('depth') 
        assert G.attributes.keys(), ['depth']
        assert allclose(G.attributes['depth'], [2, 4, 200, 400])
        assert allclose(G.get_data_points(), [[1.0, 2.1], [3.0, 5.3],
                                              [5.0, 2.1], [3.0, 50.3]])
    def test_add_with_geo (self):
        """
        Difference in Geo_reference resolved
        """
        points1 = [[1.0, 2.1], [3.0, 5.3]]
        points2 = [[5.0, 6.1], [6.0, 3.3]]
        attributes1 = [2, 4]
        attributes2 = [5, 76]
        geo_ref1= Geo_reference(55, 1.0, 2.0)
        geo_ref2 = Geo_reference(zone=55,
                                 xllcorner=0.1,
                                 yllcorner=3.0,
                                 datum='wgs84',
                                 projection='UTM',
                                 units='m')
                                
        G1 = Geospatial_data(points1, attributes1, geo_ref1)
        G2 = Geospatial_data(points2, attributes2, geo_ref2)

        #Check that absolute values are as expected
        P1 = G1.get_data_points(absolute=True)
        assert allclose(P1, [[2.0, 4.1], [4.0, 7.3]])

        P2 = G2.get_data_points(absolute=True)
        assert allclose(P2, [[5.1, 9.1], [6.1, 6.3]])        
        
        G = G1 + G2
        
        assert allclose(G.get_geo_reference().get_xllcorner(), 0.1)
        assert allclose(G.get_geo_reference().get_yllcorner(), 2.0)

        P = G.get_data_points(absolute=True)

        P_relative = G.get_data_points(absolute=False)
        
        assert allclose(P_relative, P - [0.1, 2.0])

        assert allclose(P, concatenate( (P1,P2) ))
        assert allclose(P, [[2.0, 4.1], [4.0, 7.3],
                            [5.1, 9.1], [6.1, 6.3]])
        


        

    def test_add_with_geo_absolute (self):
        """
        Difference in Geo_reference resolved
        """
        points1 = array([[2.0, 4.1], [4.0, 7.3]])
        points2 = array([[5.1, 9.1], [6.1, 6.3]])        
        attributes1 = [2, 4]
        attributes2 = [5, 76]
        geo_ref1= Geo_reference(55, 1.0, 2.0)
        geo_ref2 = Geo_reference(55, 2.0, 3.0)

        
                                
        G1 = Geospatial_data(points1 - [geo_ref1.get_xllcorner(), geo_ref1.get_yllcorner()],
                             attributes1, geo_ref1)
        
        G2 = Geospatial_data(points2 - [geo_ref2.get_xllcorner(), geo_ref2.get_yllcorner()],
                             attributes2, geo_ref2)

        #Check that absolute values are as expected
        P1 = G1.get_data_points(absolute=True)
        assert allclose(P1, points1)

        P1 = G1.get_data_points(absolute=False)
        assert allclose(P1, points1 - [geo_ref1.get_xllcorner(), geo_ref1.get_yllcorner()])        

        P2 = G2.get_data_points(absolute=True)
        assert allclose(P2, points2)

        P2 = G2.get_data_points(absolute=False)
        assert allclose(P2, points2 - [geo_ref2.get_xllcorner(), geo_ref2.get_yllcorner()])                
        
        G = G1 + G2
        
        assert allclose(G.get_geo_reference().get_xllcorner(), 1.0)
        assert allclose(G.get_geo_reference().get_yllcorner(), 2.0)

        P = G.get_data_points(absolute=True)

        P_relative = G.get_data_points(absolute=False)
        
        assert allclose(P_relative, [[1.0, 2.1], [3.0, 5.3], [4.1, 7.1], [5.1, 4.3]])

        assert allclose(P, concatenate( (points1,points2) ))
                           
        
    def test_clip0(self):
        """test_clip0(self):
        
        Test that point sets can be clipped by a polygon
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]
        G = Geospatial_data(points)

        # First try the unit square    
        U = [[0,0], [1,0], [1,1], [0,1]] 
        assert allclose(G.clip(U).get_data_points(), [[0.2, 0.5], [0.4, 0.3], [0, 0]])

        # Then a more complex polygon
        polygon = [[0,0], [1,0], [0.5,-1], [2, -1], [2,1], [0,1]]
        points = [ [0.5, 1.4], [0.5, 0.5], [1, -0.5], [1.5, 0], [0.5, 1.5], [0.5, -0.5]]
        G = Geospatial_data(points)

        assert allclose(G.clip(polygon).get_data_points(),
                        [[0.5, 0.5], [1, -0.5], [1.5, 0]])

    def test_clip0_with_attributes(self):
        """test_clip0_with_attributes(self):
        
        Test that point sets with attributes can be clipped by a polygon
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]

        attributes = [2, -4, 5, 76, -2, 0.1, 3]
        att_dict = {'att1': attributes,
                    'att2': array(attributes)+1}
        
        G = Geospatial_data(points, att_dict)

        # First try the unit square    
        U = [[0,0], [1,0], [1,1], [0,1]] 
        assert allclose(G.clip(U).get_data_points(), [[0.2, 0.5], [0.4, 0.3], [0, 0]])
        assert allclose(G.clip(U).get_attributes('att1'), [-4, 76, 0.1])
        assert allclose(G.clip(U).get_attributes('att2'), [-3, 77, 1.1])                

        # Then a more complex polygon
        polygon = [[0,0], [1,0], [0.5,-1], [2, -1], [2,1], [0,1]]
        points = [ [0.5, 1.4], [0.5, 0.5], [1, -0.5], [1.5, 0], [0.5, 1.5], [0.5, -0.5]]

        # This time just one attribute
        attributes = [2, -4, 5, 76, -2, 0.1]
        G = Geospatial_data(points, attributes)

        assert allclose(G.clip(polygon).get_data_points(),
                        [[0.5, 0.5], [1, -0.5], [1.5, 0]])
        assert allclose(G.clip(polygon).get_attributes(), [-4, 5, 76])
        

    def test_clip1(self):
        """test_clip1(self):
        
        Test that point sets can be clipped by a polygon given as
        another Geospatial dataset
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]
        attributes = [2, -4, 5, 76, -2, 0.1, 3]
        att_dict = {'att1': attributes,
                    'att2': array(attributes)+1}
        G = Geospatial_data(points, att_dict)
        
        # First try the unit square    
        U = Geospatial_data([[0,0], [1,0], [1,1], [0,1]]) 
        assert allclose(G.clip(U).get_data_points(),
                        [[0.2, 0.5], [0.4, 0.3], [0, 0]])

        assert allclose(G.clip(U).get_attributes('att1'), [-4, 76, 0.1])
        assert allclose(G.clip(U).get_attributes('att2'), [-3, 77, 1.1])                        
        
        # Then a more complex polygon
        points = [ [0.5, 1.4], [0.5, 0.5], [1, -0.5], [1.5, 0], [0.5, 1.5], [0.5, -0.5]]
        attributes = [2, -4, 5, 76, -2, 0.1]        
        G = Geospatial_data(points, attributes)
        polygon = Geospatial_data([[0,0], [1,0], [0.5,-1], [2, -1], [2,1], [0,1]])
        

        assert allclose(G.clip(polygon).get_data_points(),
                        [[0.5, 0.5], [1, -0.5], [1.5, 0]])
        assert allclose(G.clip(polygon).get_attributes(), [-4, 5, 76])
        

    def test_clip0_outside(self):
        """test_clip0_outside(self):
        
        Test that point sets can be clipped outside of a polygon
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]
        attributes = [2, -4, 5, 76, -2, 0.1, 3]        
        G = Geospatial_data(points, attributes)

        # First try the unit square    
        U = [[0,0], [1,0], [1,1], [0,1]]
        assert allclose(G.clip_outside(U).get_data_points(),
                        [[-1, 4], [1.0, 2.1], [3.0, 5.3], [2.4, 3.3]])
        #print G.clip_outside(U).get_attributes()
        assert allclose(G.clip_outside(U).get_attributes(), [2, 5, -2, 3])        
        

        # Then a more complex polygon
        polygon = [[0,0], [1,0], [0.5,-1], [2, -1], [2,1], [0,1]]
        points = [ [0.5, 1.4], [0.5, 0.5], [1, -0.5], [1.5, 0], [0.5, 1.5], [0.5, -0.5]]
        attributes = [2, -4, 5, 76, -2, 0.1]        
        G = Geospatial_data(points, attributes)

        assert allclose(G.clip_outside(polygon).get_data_points(),
                        [[0.5, 1.4], [0.5, 1.5], [0.5, -0.5]])
        assert allclose(G.clip_outside(polygon).get_attributes(), [2, -2, 0.1])                


    def test_clip1_outside(self):
        """test_clip1_outside(self):
        
        Test that point sets can be clipped outside of a polygon given as
        another Geospatial dataset
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]
        attributes = [2, -4, 5, 76, -2, 0.1, 3]        
        G = Geospatial_data(points, attributes)        

        # First try the unit square    
        U = Geospatial_data([[0,0], [1,0], [1,1], [0,1]]) 
        assert allclose(G.clip_outside(U).get_data_points(),
                        [[-1, 4], [1.0, 2.1], [3.0, 5.3], [2.4, 3.3]])
        assert allclose(G.clip(U).get_attributes(), [-4, 76, 0.1])        

        # Then a more complex polygon
        points = [ [0.5, 1.4], [0.5, 0.5], [1, -0.5], [1.5, 0], [0.5, 1.5], [0.5, -0.5]]
        attributes = [2, -4, 5, 76, -2, 0.1]        
        G = Geospatial_data(points, attributes)

        polygon = Geospatial_data([[0,0], [1,0], [0.5,-1], [2, -1], [2,1], [0,1]])
        

        assert allclose(G.clip_outside(polygon).get_data_points(),
                        [[0.5, 1.4], [0.5, 1.5], [0.5, -0.5]])
        assert allclose(G.clip_outside(polygon).get_attributes(), [2, -2, 0.1])
        


    def test_clip1_inside_outside(self):
        """test_clip1_inside_outside(self):
        
        Test that point sets can be clipped outside of a polygon given as
        another Geospatial dataset
        """
        
        from anuga.coordinate_transforms.geo_reference import Geo_reference
        
        points = [[-1, 4], [0.2, 0.5], [1.0, 2.1], [0.4, 0.3], [3.0, 5.3],
                  [0, 0], [2.4, 3.3]]
        attributes = [2, -4, 5, 76, -2, 0.1, 3]        
        G = Geospatial_data(points, attributes)

        # First try the unit square    
        U = Geospatial_data([[0,0], [1,0], [1,1], [0,1]]) 
        G1 = G.clip(U)
        assert allclose(G1.get_data_points(),[[0.2, 0.5], [0.4, 0.3], [0, 0]])
        assert allclose(G.clip(U).get_attributes(), [-4, 76, 0.1])
        
        G2 = G.clip_outside(U)
        assert allclose(G2.get_data_points(),[[-1, 4], [1.0, 2.1],
                                              [3.0, 5.3], [2.4, 3.3]])
        assert allclose(G.clip_outside(U).get_attributes(), [2, 5, -2, 3])                

        
        # New ordering
        new_points = [[0.2, 0.5], [0.4, 0.3], [0, 0]] +\
                     [[-1, 4], [1.0, 2.1], [3.0, 5.3], [2.4, 3.3]]

        new_attributes = [-4, 76, 0.1, 2, 5, -2, 3]                 
        
        assert allclose((G1+G2).get_data_points(), new_points)
        assert allclose((G1+G2).get_attributes(), new_attributes)

        G = G1+G2
        FN = 'test_combine.pts'
        G.export_points_file(FN)


        # Read it back in
        G3 = Geospatial_data(FN)


        # Check result
        assert allclose(G3.get_data_points(), new_points)        
        assert allclose(G3.get_attributes(), new_attributes)        
        
        os.remove(FN)

        
    def test_create_from_xya_file(self):
        """Check that object can be created from a points file (.pts and .xya)
        """

        points = [[1.0, 2.1], [3.0, 5.3], [5.0, 6.1], [6.0, 3.3]]
        attributes = [2, 4, 5, 76]
        '''
        # Use old pointsdict format
        pointsdict = {'pointlist': points,
                      'attributelist': {'att1': attributes,
                                        'att2': array(attributes) + 1}} 
        '''
        att_dict = {'att1': attributes,
                    'att2': array(attributes) +1}
                    
        # Create points as an xya file 
        FN = 'test_points.xya'
        G1 = Geospatial_data(points, att_dict)
        G1.export_points_file(FN)
#        G1.export_points_file(ofile)

        #Create object from file
        G = Geospatial_data(file_name = FN)
        
        assert allclose(G.get_data_points(), points)
        assert allclose(G.get_attributes('att1'), attributes)
        assert allclose(G.get_attributes('att2'), array(attributes) + 1)
        
        os.remove(FN)

    def test_create_from_xya_file1(self):
        """
        Check that object can be created from an Absolute xya file
        """

        points = [[1.0, 2.1], [3.0, 5.3], [5.0, 6.1], [6.0, 3.3]]
        attributes = [2, 4, 5, 76]

        att_dict = {'att1': attributes,
                    'att2': array(attributes) +1}

        geo_ref = Geo_reference(56, 10, 5)
                    
        # Create points as an xya file 
        FN = 'test_points.xya'
        G1 = Geospatial_data(points, att_dict, geo_ref)

        G1.export_points_file(FN, absolute=True)

        #Create object from file
        G = Geospatial_data(file_name = FN)
        
        assert allclose(G.get_data_points(absolute=True), 
                        G1.get_data_points(absolute=True))
        assert allclose(G.get_attributes('att1'), attributes)
        assert allclose(G.get_attributes('att2'), array(attributes) + 1)
        
        os.remove(FN)
    
    def test_load_csv(self):
        
        import os
        import tempfile
       
        fileName = tempfile.mktemp(".csv")
        file = open(fileName,"w")
        file.write("x,y,elevation speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n")
        file.close()
        #print fileName
        results = Geospatial_data(fileName, delimiter=',')
        os.remove(fileName)
#        print 'data', results.get_data_points()
        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],
                                                    [1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'),
                        [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'),
                        [0.0, 10.0, 40.0])

        
    def test_load_xya(self):
        """ test_load_xya(self):
        comma delimited
        """
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("elevation  , speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.4, 40.0\n")
        file.close()
        results = Geospatial_data(fileName, delimiter=',')
        os.remove(fileName)
#        print 'data', results.get_data_points()
        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), [0.0, 10.0, 40.0])

    def test_loadxya2(self):
        """ test_loadxya2(self):
        space delimited
        """
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n")
        file.close()

        results = Geospatial_data(fileName, delimiter=' ')

        os.remove(fileName)

        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), [0.0, 10.0, 40.0])
     
    def test_loadxya3(self):
        """
        test_loadxya3(self):
        space delimited
        """
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n\
#geocrap\n\
56\n\
56.6\n\
3\n")
        file.close()

        results = Geospatial_data(fileName, delimiter=' ')

        os.remove(fileName)
        assert allclose(results.get_data_points(absolute=False), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), [0.0, 10.0, 40.0])

    def BADtest_loadxya4(self):
        """
        comma delimited
        """
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("elevation  , speed \n\
1.0, 0.0, splat, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.4, 40.0\n")
        file.close()
        results = Geospatial_data(fileName, delimiter=',')
        os.remove(fileName)
#        print 'data', results.get_data_points()
        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), ["splat", 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), [0.0, 10.0, 40.0])
        
    def test_read_write_points_file_bad2(self):
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference=Geo_reference(56,1.9,1.9)
        
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        
        try:
            G.export_points_file("_???/yeah.xya")
            
        except IOError:
            pass
        else:
            msg = 'bad points file extension did not raise error!'
            raise msg
#            self.failUnless(0 == 1,
#                        'bad points file extension did not raise error!')
                   
    def test_loadxy_bad(self):
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n")
        file.close()
        #print fileName
        try:
            results = Geospatial_data(fileName, delimiter=' ')
        except IOError:
            pass
        else:
            msg = 'bad xya file did not raise error!'
            raise msg
#            self.failUnless(0 == 1,
#                        'bad xya file did not raise error!')
        os.remove(fileName)
       
    def test_loadxy_bad2(self):
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("elevation\n\
1.0 0.0 10.0 \n\
0.0 1.0\n\
1.0 \n")
        file.close()
        #print fileName
        try:
            results = Geospatial_data(fileName, delimiter=' ')
        except IOError:
            pass
        else:
            msg = 'bad xya file did not raise error!'
            raise msg
        os.remove(fileName)
   
    def test_loadxy_bad3(self):
        """ test_loadxy_bad3(self):
        specifying wrong delimiter
        """
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation  , speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.4, 40.0\n")
        file.close()
        try:
            results = Geospatial_data(fileName, delimiter=' ')
        except IOError:
            pass
        else:
            msg = 'bad xya file did not raise error!'
            raise msg
        os.remove(fileName)
     
    def test_loadxy_bad4(self):
        """ test_loadxy_bad4(self):
         specifying wrong delimiter
        """
        import os
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n\
#geocrap\n\
56\n\
56.6\n\
3\n"
)
        file.close()
        try:
            results = Geospatial_data(fileName, delimiter=',')
        except IOError:
            pass
        else:
            msg = 'bad xya file did not raise error!'
            raise msg

        os.remove(fileName)

    def test_loadxy_bad5(self):
        """ test_loadxy_bad5(self):
        specifying wrong delimiter
        """
        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n\
#geocrap\n\
crap")
        file.close()
        try:
#            dict = import_points_file(fileName,delimiter=' ')
#            results = Geospatial_data()
            results = Geospatial_data(fileName, delimiter=' ', verbose=True)
#            results.import_points_file(fileName, delimiter=' ')
        except IOError:
            pass
        else:
            msg = 'bad xya file did not raise error!'
            raise msg

#            self.failUnless(0 ==1,
#                        'bad xya file did not raise error!')    
        os.remove(fileName)              

    def test_loadxy_bad_no_file_xya(self):
        import os
       
        fileName = tempfile.mktemp(".xya")
        try:
            results = Geospatial_data(fileName, delimiter=' ')
        except IOError:
            pass
        else:
            msg = 'imaginary file did not raise error!'
            raise msg

#        except IOError:
#            pass
#        else:
#            self.failUnless(0 == 1,
#                        'imaginary file did not raise error!')


  ###################### .CSV ##############################

    def test_load_csv_lat_long_bad_blocking(self):
        
        fileName = tempfile.mktemp(".csv")
        file = open(fileName,"w")
        file.write("Lati,LONG,z \n\
-25.0,180.0,452.688000\n\
-34,150.0,459.126000\n")
        file.close()
        
        results = Geospatial_data(fileName, max_read_lines=1,
                                  load_file_now=False)
        
        try:
            for i in results:
                pass
        except ANUGAError:
            pass
        else:
            msg = 'Different zones in Geo references not caught.'
            raise msg        
        
        os.remove(fileName)
        
    def test_load_csv(self):
        
        fileName = tempfile.mktemp(".txt")
        file = open(fileName,"w")
        file.write(" x,y, elevation ,  speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0 ,10.4, 40.0\n")
        file.close()

        results = Geospatial_data(fileName, max_read_lines=2)


        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), [0.0, 10.0, 40.0])

        # Blocking
        geo_list = []
        for i in results:
            geo_list.append(i)
            
        assert allclose(geo_list[0].get_data_points(),
                        [[1.0, 0.0],[0.0, 1.0]])

        assert allclose(geo_list[0].get_attributes(attribute_name='elevation'),
                        [10.0, 0.0])
        assert allclose(geo_list[1].get_data_points(),
                        [[1.0, 0.0]])        
        assert allclose(geo_list[1].get_attributes(attribute_name='elevation'),
                        [10.4])
           
        os.remove(fileName)         
        
    def test_load_csv_bad(self):
        """ test_load_csv_bad(self):
        space delimited
        """
        import os
       
        fileName = tempfile.mktemp(".txt")
        file = open(fileName,"w")
        file.write(" elevation ,  speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0 ,10.4, 40.0\n")
        file.close()

        results = Geospatial_data(fileName, max_read_lines=2,
                                  load_file_now=False)

        # Blocking
        geo_list = []
        try:
            for i in results:
                geo_list.append(i)
        except IOError:
            pass
        else:
            msg = 'bad file did not raise error!'
            raise msg
        os.remove(fileName)

        
    def test_load_pts_blocking(self):
        """ test_load_csv(self):
        space delimited
        """
        import os
       
        fileName = tempfile.mktemp(".txt")
        file = open(fileName,"w")
        file.write(" x,y, elevation ,  speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0 ,10.4, 40.0\n")
        file.close()

        pts_file = tempfile.mktemp(".pts")
        
        convert = Geospatial_data(fileName)
        convert.export_points_file(pts_file)
        results = Geospatial_data(pts_file, max_read_lines=2)

        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],
                                                    [1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'),
                        [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'),
                        [0.0, 10.0, 40.0])

        # Blocking
        geo_list = []
        for i in results:
            geo_list.append(i) 
        assert allclose(geo_list[0].get_data_points(),
                        [[1.0, 0.0],[0.0, 1.0]])
        assert allclose(geo_list[0].get_attributes(attribute_name='elevation'),
                        [10.0, 0.0])
        assert allclose(geo_list[1].get_data_points(),
                        [[1.0, 0.0]])        
        assert allclose(geo_list[1].get_attributes(attribute_name='elevation'),
                        [10.4])
           
        os.remove(fileName)  
        os.remove(pts_file)               

    def verbose_test_load_pts_blocking(self):
        
        import os
       
        fileName = tempfile.mktemp(".txt")
        file = open(fileName,"w")
        file.write(" x,y, elevation ,  speed \n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0, 10.0, 0.0\n\
0.0, 1.0, 0.0, 10.0\n\
1.0, 0.0 ,10.4, 40.0\n")
        file.close()

        pts_file = tempfile.mktemp(".pts")
        
        convert = Geospatial_data(fileName)
        convert.export_points_file(pts_file)
        results = Geospatial_data(pts_file, max_read_lines=2, verbose=True)

        # Blocking
        geo_list = []
        for i in results:
            geo_list.append(i) 
        assert allclose(geo_list[0].get_data_points(),
                        [[1.0, 0.0],[0.0, 1.0]])
        assert allclose(geo_list[0].get_attributes(attribute_name='elevation'),
                        [10.0, 0.0])
        assert allclose(geo_list[1].get_data_points(),
                        [[1.0, 0.0],[0.0, 1.0] ])        
        assert allclose(geo_list[1].get_attributes(attribute_name='elevation'),
                        [10.0, 0.0])
           
        os.remove(fileName)  
        os.remove(pts_file)
        
        
    def test_export_xya_file(self):
        # depreciated since it's testing using geo refs in a text file
#        dict = {}
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
#        dict['attributelist'] = att_dict
        geo_reference=Geo_reference(56,1.9,1.9)
        
        
        fileName = tempfile.mktemp(".xya")
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        G.export_points_file(fileName, False)

#        dict2 = import_points_file(fileName)
        results = Geospatial_data(file_name = fileName)
        #print "fileName",fileName 
        os.remove(fileName)
        
        assert allclose(results.get_data_points(absolute=False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes(attribute_name='brightness'), answer)
        #print "dict2['geo_reference']",dict2['geo_reference'] 
        self.failUnless(results.get_geo_reference() == geo_reference,
                         'test_writepts failed. Test geo_reference')

    def test_export_xya_file2(self):
        """ test_export_xya_file2(self):
        test absolute xya file
        """
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        
        fileName = tempfile.mktemp(".xya")
        G = Geospatial_data(pointlist, att_dict)
        G.export_points_file(fileName)
        results = Geospatial_data(file_name = fileName)
#        dict2 = import_points_file(fileName)
        os.remove(fileName)
        
        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes('elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes('brightness'), answer)

    def test_export_xya_file3(self):
        """ test_export_xya_file3(self):
        test absolute xya file with geo_ref
        """
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference=Geo_reference(56,1.9,1.9)
        
        
        fileName = tempfile.mktemp(".xya")
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        
        G.export_points_file(fileName, absolute=True)
        
        results = Geospatial_data(file_name = fileName)
        os.remove(fileName)

        assert allclose(results.get_data_points(),
                        [[2.9, 1.9],[1.9, 2.9],[2.9, 1.9]])
        assert allclose(results.get_attributes(attribute_name='elevation'),
                         [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes(attribute_name='brightness'), answer)
        self.failUnless(results.get_geo_reference() == geo_reference,
                         'test_writepts failed. Test geo_reference')                         
                        
                        
                        
    def test_new_export_pts_file(self):
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.1, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 1.0, 10.4])
        
        fileName = tempfile.mktemp(".pts")
        
        G = Geospatial_data(pointlist, att_dict)
        
        G.export_points_file(fileName)

        results = Geospatial_data(file_name = fileName)

        os.remove(fileName)
        
        assert allclose(results.get_data_points(),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.1, 0.0, 10.4])
        answer = [10.0, 1.0, 10.4]
        assert allclose(results.get_attributes(attribute_name='brightness'), answer)

    def test_new_export_absolute_pts_file(self):
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.1, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 1.0, 10.4])
        geo_ref = Geo_reference(50, 25, 55)
        
        fileName = tempfile.mktemp(".pts")
        
        G = Geospatial_data(pointlist, att_dict, geo_ref)
        
        G.export_points_file(fileName, absolute=True)

        results = Geospatial_data(file_name = fileName)

        os.remove(fileName)
        
        assert allclose(results.get_data_points(), G.get_data_points(True))
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.1, 0.0, 10.4])
        answer = [10.0, 1.0, 10.4]
        assert allclose(results.get_attributes(attribute_name='brightness'), answer)

    def test_loadpts(self):
        
        from Scientific.IO.NetCDF import NetCDFFile

        fileName = tempfile.mktemp(".pts")
        # NetCDF file definition
        outfile = NetCDFFile(fileName, 'w')
        
        # dimension definitions
        outfile.createDimension('number_of_points', 3)    
        outfile.createDimension('number_of_dimensions', 2) #This is 2d data
    
        # variable definitions
        outfile.createVariable('points', Float, ('number_of_points',
                                                 'number_of_dimensions'))
        outfile.createVariable('elevation', Float, ('number_of_points',))
    
        # Get handles to the variables
        points = outfile.variables['points']
        elevation = outfile.variables['elevation']
 
        points[0, :] = [1.0,0.0]
        elevation[0] = 10.0 
        points[1, :] = [0.0,1.0]
        elevation[1] = 0.0  
        points[2, :] = [1.0,0.0]
        elevation[2] = 10.4    

        outfile.close()
        
        results = Geospatial_data(file_name = fileName)
        os.remove(fileName)
        answer =  [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]]
        assert allclose(results.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), [10.0, 0.0, 10.4])
        
    def test_writepts(self):
        #test_writepts: Test that storage of x,y,attributes works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference=Geo_reference(56,1.9,1.9)

        # Test pts format
        fileName = tempfile.mktemp(".pts")
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        G.export_points_file(fileName, False)
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)

        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes('elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes('brightness'), answer)
        self.failUnless(geo_reference == geo_reference,
                         'test_writepts failed. Test geo_reference')

    def test_write_xya_attributes(self):
        #test_write xya: Test that storage of x,y,attributes works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference=Geo_reference(56,0,0)
        # Test xya format
        fileName = tempfile.mktemp(".xya")
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        G.export_points_file(fileName)
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)
        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes('elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes('brightness'), answer)
        self.failUnless(geo_reference == geo_reference,
                         'test_writepts failed. Test geo_reference')

    def test_write_csv_attributes(self):
        #test_write : Test that storage of x,y,attributes works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference=Geo_reference(56,0,0)
        # Test xya format
        fileName = tempfile.mktemp(".txt")
        G = Geospatial_data(pointlist, att_dict, geo_reference)
        G.export_points_file(fileName)
        #print "fileName", fileName 
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)
        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes('elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes('brightness'), answer)
        
 
    def test_write_csv_attributes_lat_long(self):
        #test_write : Test that storage of x,y,attributes works
        
        att_dict = {}
        pointlist = array([[-21.5,114.5],[-21.6,114.5],[-21.7,114.5]])
        att_dict['elevation'] = array([10.0, 0.0, 10.4])
        att_dict['brightness'] = array([10.0, 0.0, 10.4])
        # Test txt format
        fileName = tempfile.mktemp(".txt")
        G = Geospatial_data(pointlist, att_dict, points_are_lats_longs=True)
        G.export_points_file(fileName, as_lat_long=True)
        #print "fileName", fileName 
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)
        assert allclose(results.get_data_points(False, as_lat_long=True),
                        pointlist)
        assert allclose(results.get_attributes('elevation'), [10.0, 0.0, 10.4])
        answer = [10.0, 0.0, 10.4]
        assert allclose(results.get_attributes('brightness'), answer)
        
    def test_writepts_no_attributes(self):

        #test_writepts_no_attributes: Test that storage of x,y alone works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        geo_reference=Geo_reference(56,1.9,1.9)

        # Test pts format
        fileName = tempfile.mktemp(".pts")
        G = Geospatial_data(pointlist, None, geo_reference)
        G.export_points_file(fileName, False)
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)

        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        self.failUnless(geo_reference == geo_reference,
                         'test_writepts failed. Test geo_reference')
        
    def test_write_xya_no_attributes(self):
        #test_write xya _no_attributes: Test that storage of x,y alone works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        geo_reference=Geo_reference(56,0,0)
        # Test xya format
        fileName = tempfile.mktemp(".xya")
        G = Geospatial_data(pointlist, None, geo_reference)
        G.export_points_file(fileName)
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)
        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        self.failUnless(geo_reference == geo_reference,
                         'test_writepts failed. Test geo_reference')

       
    def test_write_csv_no_attributes(self):
        #test_write xya _no_attributes: Test that storage of x,y alone works
        
        att_dict = {}
        pointlist = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        geo_reference=Geo_reference(56,0,0)
        # Test xya format
        fileName = tempfile.mktemp(".txt")
        G = Geospatial_data(pointlist, None, geo_reference)
        G.export_points_file(fileName)
        results = Geospatial_data(file_name=fileName)
        os.remove(fileName)
        assert allclose(results.get_data_points(False),[[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])

         
        
 ########################## BAD .PTS ##########################          

    def test_load_bad_no_file_pts(self):
        import os
        import tempfile
       
        fileName = tempfile.mktemp(".pts")
        #print fileName
        try:
            results = Geospatial_data(file_name = fileName)
#            dict = import_points_file(fileName)
        except IOError:
            pass
        else:
            msg = 'imaginary file did not raise error!'
            raise msg
#            self.failUnless(0 == 1,
#                        'imaginary file did not raise error!')


    def test_create_from_pts_file(self):
        
        from Scientific.IO.NetCDF import NetCDFFile

#        fileName = tempfile.mktemp(".pts")
        FN = 'test_points.pts'
        # NetCDF file definition
        outfile = NetCDFFile(FN, 'w')
        
        # dimension definitions
        outfile.createDimension('number_of_points', 3)    
        outfile.createDimension('number_of_dimensions', 2) #This is 2d data
    
        # variable definitions
        outfile.createVariable('points', Float, ('number_of_points',
                                                 'number_of_dimensions'))
        outfile.createVariable('elevation', Float, ('number_of_points',))
    
        # Get handles to the variables
        points = outfile.variables['points']
        elevation = outfile.variables['elevation']
 
        points[0, :] = [1.0,0.0]
        elevation[0] = 10.0 
        points[1, :] = [0.0,1.0]
        elevation[1] = 0.0  
        points[2, :] = [1.0,0.0]
        elevation[2] = 10.4    

        outfile.close()

        G = Geospatial_data(file_name = FN)

        assert allclose(G.get_geo_reference().get_xllcorner(), 0.0)
        assert allclose(G.get_geo_reference().get_yllcorner(), 0.0)

        assert allclose(G.get_data_points(), [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(G.get_attributes(), [10.0, 0.0, 10.4])
        os.remove(FN)

    def test_create_from_pts_file_with_geo(self):
        """This test reveals if Geospatial data is correctly instantiated from a pts file.
        """
        
        from Scientific.IO.NetCDF import NetCDFFile

        FN = 'test_points.pts'
        # NetCDF file definition
        outfile = NetCDFFile(FN, 'w')

        # Make up an arbitrary georef
        xll = 0.1
        yll = 20
        geo_reference=Geo_reference(56, xll, yll)
        geo_reference.write_NetCDF(outfile)

        # dimension definitions
        outfile.createDimension('number_of_points', 3)    
        outfile.createDimension('number_of_dimensions', 2) #This is 2d data
    
        # variable definitions
        outfile.createVariable('points', Float, ('number_of_points',
                                                 'number_of_dimensions'))
        outfile.createVariable('elevation', Float, ('number_of_points',))
    
        # Get handles to the variables
        points = outfile.variables['points']
        elevation = outfile.variables['elevation']

        points[0, :] = [1.0,0.0]
        elevation[0] = 10.0 
        points[1, :] = [0.0,1.0]
        elevation[1] = 0.0  
        points[2, :] = [1.0,0.0]
        elevation[2] = 10.4    

        outfile.close()

        G = Geospatial_data(file_name = FN)

        assert allclose(G.get_geo_reference().get_xllcorner(), xll)
        assert allclose(G.get_geo_reference().get_yllcorner(), yll)

        assert allclose(G.get_data_points(), [[1.0+xll, 0.0+yll],
                                              [0.0+xll, 1.0+yll],
                                              [1.0+xll, 0.0+yll]])
        
        assert allclose(G.get_attributes(), [10.0, 0.0, 10.4])
        os.remove(FN)

        
    def test_add_(self):
        '''test_add_(self):
        adds an xya and pts files, reads the files and adds them
           checking results are correct
        '''

        # create files
        att_dict1 = {}
        pointlist1 = array([[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        att_dict1['elevation'] = array([-10.0, 0.0, 10.4])
        att_dict1['brightness'] = array([10.0, 0.0, 10.4])
        geo_reference1 = Geo_reference(56, 2.0, 1.0)
        
        att_dict2 = {}
        pointlist2 = array([[2.0, 1.0],[1.0, 2.0],[2.0, 1.0]])
        att_dict2['elevation'] = array([1.0, 15.0, 1.4])
        att_dict2['brightness'] = array([14.0, 1.0, -12.4])
        geo_reference2 = Geo_reference(56, 1.0, 2.0) 

        G1 = Geospatial_data(pointlist1, att_dict1, geo_reference1)
        G2 = Geospatial_data(pointlist2, att_dict2, geo_reference2)
        
        fileName1 = tempfile.mktemp(".xya")
        fileName2 = tempfile.mktemp(".pts")

        #makes files
        G1.export_points_file(fileName1)
        G2.export_points_file(fileName2)
        
        # add files
        
        G3 = Geospatial_data(file_name = fileName1)
        G4 = Geospatial_data(file_name = fileName2)
        
        G = G3 + G4

        
        #read results
#        print'res', G.get_data_points()
#        print'res1', G.get_data_points(False)
        assert allclose(G.get_data_points(),
                        [[ 3.0, 1.0], [ 2.0, 2.0],
                         [ 3.0, 1.0], [ 3.0, 3.0],
                         [ 2.0, 4.0], [ 3.0, 3.0]])
                         
        assert allclose(G.get_attributes(attribute_name='elevation'),
                        [-10.0, 0.0, 10.4, 1.0, 15.0, 1.4])
        
        answer = [10.0, 0.0, 10.4, 14.0, 1.0, -12.4]
        assert allclose(G.get_attributes(attribute_name='brightness'), answer)
        
        self.failUnless(G.get_geo_reference() == geo_reference1,
                         'test_writepts failed. Test geo_reference')
                         
        os.remove(fileName1)
        os.remove(fileName2)
        
    def test_ensure_absolute(self):
        points = [[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]]
        new_points = ensure_absolute(points)
        
        assert allclose(new_points, points)

        points = array([[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]])
        new_points = ensure_absolute(points)
        
        assert allclose(new_points, points)
        
        ab_points = array([[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]])
        
        mesh_origin = (56, 290000, 618000) #zone, easting, northing

        data_points = zeros((ab_points.shape), Float)
        #Shift datapoints according to new origins
        for k in range(len(ab_points)):
            data_points[k][0] = ab_points[k][0] - mesh_origin[1]
            data_points[k][1] = ab_points[k][1] - mesh_origin[2]
        #print "data_points",data_points     
        new_points = ensure_absolute(data_points,
                                             geo_reference=mesh_origin)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)

        geo = Geo_reference(56,67,-56)

        data_points = geo.change_points_geo_ref(ab_points)   
        new_points = ensure_absolute(data_points,
                                             geo_reference=geo)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)


        geo_reference = Geo_reference(56, 100, 200)
        ab_points = [[1.0, 2.1], [3.0, 5.3]]
        points = geo_reference.change_points_geo_ref(ab_points)
        attributes = [2, 4]
        #print "geo in points", points
        G = Geospatial_data(points, attributes,
                            geo_reference=geo_reference)
          
        new_points = ensure_absolute(G)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)

       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n\
#geocrap\n\
56\n\
10\n\
20\n")
        file.close()
        
        ab_points = ensure_absolute(fileName)
        actual =  [[11, 20.0],[10.0, 21.0],[11.0, 20.0]]
        assert allclose(ab_points, actual)
        os.remove(fileName)

        
    def test_ensure_geospatial(self):
        points = [[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]]
        new_points = ensure_geospatial(points)
        
        assert allclose(new_points.get_data_points(absolute = True), points)

        points = array([[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]])
        new_points = ensure_geospatial(points)
        
        assert allclose(new_points.get_data_points(absolute = True), points)
        
        ab_points = array([[2.0, 0.0],[1.0, 1.0],
                         [2.0, 0.0],[2.0, 2.0],
                         [1.0, 3.0],[2.0, 2.0]])
        
        mesh_origin = (56, 290000, 618000) #zone, easting, northing

        data_points = zeros((ab_points.shape), Float)
        #Shift datapoints according to new origins
        for k in range(len(ab_points)):
            data_points[k][0] = ab_points[k][0] - mesh_origin[1]
            data_points[k][1] = ab_points[k][1] - mesh_origin[2]
        #print "data_points",data_points     
        new_geospatial = ensure_geospatial(data_points,
                                             geo_reference=mesh_origin)
        new_points = new_geospatial.get_data_points(absolute=True)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)

        geo = Geo_reference(56,67,-56)

        data_points = geo.change_points_geo_ref(ab_points)   
        new_geospatial = ensure_geospatial(data_points,
                                             geo_reference=geo)
        new_points = new_geospatial.get_data_points(absolute=True)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)


        geo_reference = Geo_reference(56, 100, 200)
        ab_points = [[1.0, 2.1], [3.0, 5.3]]
        points = geo_reference.change_points_geo_ref(ab_points)
        attributes = [2, 4]
        #print "geo in points", points
        G = Geospatial_data(points, attributes,
                            geo_reference=geo_reference)
          
        new_geospatial  = ensure_geospatial(G)
        new_points = new_geospatial.get_data_points(absolute=True)
        #print "new_points",new_points
        #print "ab_points",ab_points
           
        assert allclose(new_points, ab_points)
        
    def test_isinstance(self):

        import os
       
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation   speed \n\
1.0 0.0 10.0 0.0\n\
0.0 1.0 0.0 10.0\n\
1.0 0.0 10.4 40.0\n\
#geocrap\n\
56\n\
56.6\n\
3\n")
        file.close()

        results = Geospatial_data(fileName)
        assert allclose(results.get_data_points(absolute=False), \
                        [[1.0, 0.0],[0.0, 1.0],[1.0, 0.0]])
        assert allclose(results.get_attributes(attribute_name='elevation'), \
                        [10.0, 0.0, 10.4])
        assert allclose(results.get_attributes(attribute_name='speed'), \
                        [0.0, 10.0, 40.0])

        os.remove(fileName)
        
    def test_delimiter(self):
        
        try:
            G = Geospatial_data(delimiter=',')
#            results = Geospatial_data(file_name = fileName)
#            dict = import_points_file(fileName)
        except ValueError:
            pass
        else:
            msg = 'Instance with No fileName but has a delimiter\
                  did not raise error!'
            raise msg

    def test_no_constructors(self):
        
        try:
            G = Geospatial_data()
#            results = Geospatial_data(file_name = fileName)
#            dict = import_points_file(fileName)
        except ValueError:
            pass
        else:
            msg = 'Instance must have a filename or data points'
            raise msg        

    def test_check_geo_reference(self):
        """
        checks geo reference details are OK. eg can be called '#geo reference'
        if not throws a clear error message
        """
        import os
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation  \n\
1.0 0.0 10.0\n\
0.0 1.0 0.0\n\
1.0 0.0 10.4\n\
#ge oreference\n\
56\n\
1.1\n\
1.0\n")

        file.close()
        results = Geospatial_data(fileName)
        assert allclose(results.get_geo_reference().get_xllcorner(), 1.1)
        assert allclose(results.get_geo_reference().get_yllcorner(), 1.0)

        os.remove(fileName)
        
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation  \n\
1.0 0.0 10.0\n\
0.0 1.0 0.0\n\
1.0 0.0 10.4\n")

        file.close()
        results = Geospatial_data(fileName)
       
        os.remove(fileName)
        
    def test_check_geo_reference1(self):
        """
        checks geo reference details are OK. eg can be called '#geo reference'
        if not throws a clear error message
        """
        import os
        fileName = tempfile.mktemp(".xya")
        file = open(fileName,"w")
        file.write("  elevation  \n\
1.0 0.0 10.0\n\
0.0 1.0 0.0\n\
1.0 0.0 10.4\n\
#geo t t\n\
56\n\
1.1\n"
)
        file.close()

        try:
            results = Geospatial_data(fileName, delimiter = " ")
        except IOError:
            pass
        else:
            msg = 'Geo reference data format is incorrect'
            raise msg        


        os.remove(fileName)


    def test_load_csv_lat_long(self):
        """ 
        comma delimited

        """
        fileName = tempfile.mktemp(".csv")
        file = open(fileName,"w")
        file.write("long,lat, elevation, yeah \n\
150.916666667,-34.50,452.688000, 10\n\
150.0,-34,459.126000, 10\n")
        file.close()
        results = Geospatial_data(fileName, delimiter=',')
        os.remove(fileName)
        points = results.get_data_points()
        
        assert allclose(points[0][0], 308728.009)
        assert allclose(points[0][1], 6180432.601)
        assert allclose(points[1][0],  222908.705)
        assert allclose(points[1][1], 6233785.284)
        
      
    def test_load_csv_lat_longII(self):
        """ 
        comma delimited

        """
        fileName = tempfile.mktemp(".csv")
        file = open(fileName,"w")
        file.write("Lati,LONG,z \n\
-34.50,150.916666667,452.688000\n\
-34,150.0,459.126000\n")
        file.close()
        results = Geospatial_data(fileName, delimiter=',')
        os.remove(fileName)
        points = results.get_data_points()
        
        assert allclose(points[0][0], 308728.009)
        assert allclose(points[0][1], 6180432.601)
        assert allclose(points[1][0],  222908.705)
        assert allclose(points[1][1], 6233785.284)

          
    def test_load_csv_lat_long_bad(self):
        """ 
        comma delimited

        """
        fileName = tempfile.mktemp(".csv")
        file = open(fileName,"w")
        file.write("Lati,LONG,z \n\
-25.0,180.0,452.688000\n\
-34,150.0,459.126000\n")
        file.close()
        try:
            results = Geospatial_data(fileName, delimiter=',')
        except ANUGAError:
            pass
        else:
            msg = 'Different zones in Geo references not caught.'
            raise msg        
        
        os.remove(fileName)
        
    def test_lat_long(self):
        lat_gong = degminsec2decimal_degrees(-34,30,0.)
        lon_gong = degminsec2decimal_degrees(150,55,0.)
        
        lat_2 = degminsec2decimal_degrees(-34,00,0.)
        lon_2 = degminsec2decimal_degrees(150,00,0.)
        
        lats = [lat_gong, lat_2]
        longs = [lon_gong, lon_2]
        gsd = Geospatial_data(latitudes=lats, longitudes=longs)

        points = gsd.get_data_points(absolute=True)
        
        assert allclose(points[0][0], 308728.009)
        assert allclose(points[0][1], 6180432.601)
        assert allclose(points[1][0],  222908.705)
        assert allclose(points[1][1], 6233785.284)
        self.failUnless(gsd.get_geo_reference().get_zone() == 56,
                        'Bad zone error!')
        
        try:
            results = Geospatial_data(latitudes=lats)
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1,  'Error not thrown error!')
        try:
            results = Geospatial_data(latitudes=lats)
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1,  'Error not thrown error!')
        try:
            results = Geospatial_data(longitudes=lats)
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1, 'Error not thrown error!')
        try:
            results = Geospatial_data(latitudes=lats, longitudes=longs,
                                      geo_reference="p")
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1,  'Error not thrown error!')
            
        try:
            results = Geospatial_data(latitudes=lats, longitudes=longs,
                                      data_points=12)
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1,  'Error not thrown error!')

    def test_lat_long2(self):
        lat_gong = degminsec2decimal_degrees(-34,30,0.)
        lon_gong = degminsec2decimal_degrees(150,55,0.)
        
        lat_2 = degminsec2decimal_degrees(-34,00,0.)
        lon_2 = degminsec2decimal_degrees(150,00,0.)
        
        points = [[lat_gong, lon_gong], [lat_2, lon_2]]
        gsd = Geospatial_data(data_points=points, points_are_lats_longs=True)

        points = gsd.get_data_points(absolute=True)
        
        assert allclose(points[0][0], 308728.009)
        assert allclose(points[0][1], 6180432.601)
        assert allclose(points[1][0],  222908.705)
        assert allclose(points[1][1], 6233785.284)
        self.failUnless(gsd.get_geo_reference().get_zone() == 56,
                        'Bad zone error!')

        try:
            results = Geospatial_data(points_are_lats_longs=True)
        except ValueError:
            pass
        else:
            self.failUnless(0 ==1,  'Error not thrown error!')


    def test_write_urs_file(self):
        lat_gong = degminsec2decimal_degrees(-34,00,0)
        lon_gong = degminsec2decimal_degrees(150,30,0.)
        
        lat_2 = degminsec2decimal_degrees(-34,00,1)
        lon_2 = degminsec2decimal_degrees(150,00,0.)
        p1 = (lat_gong, lon_gong)
        p2 = (lat_2, lon_2)
        points = ImmutableSet([p1, p2, p1])
        gsd = Geospatial_data(data_points=list(points),
                              points_are_lats_longs=True)
        
        fn = tempfile.mktemp(".urs")
        gsd.export_points_file(fn)
        #print "fn", fn 
        handle = open(fn)
        lines = handle.readlines()
        assert lines[0],'2'
        assert lines[1],'-34.0002778 150.0 1'
        assert lines[2],'-34.0 150.5 2'
        handle.close()
        os.remove(fn)
        
    def test_lat_long_set(self):
        lat_gong = degminsec2decimal_degrees(-34,30,0.)
        lon_gong = degminsec2decimal_degrees(150,55,0.)
        
        lat_2 = degminsec2decimal_degrees(-34,00,0.)
        lon_2 = degminsec2decimal_degrees(150,00,0.)
        p1 = (lat_gong, lon_gong)
        p2 = (lat_2, lon_2)
        points = ImmutableSet([p1, p2, p1])
        gsd = Geospatial_data(data_points=list(points),
                              points_are_lats_longs=True)

        points = gsd.get_data_points(absolute=True)
        #print "points[0][0]", points[0][0]
        #Note the order is unknown, due to using sets
        # and it changes from windows to linux
        try:
            assert allclose(points[1][0], 308728.009)
            assert allclose(points[1][1], 6180432.601)
            assert allclose(points[0][0],  222908.705)
            assert allclose(points[0][1], 6233785.284)
        except AssertionError:
            assert allclose(points[0][0], 308728.009)
            assert allclose(points[0][1], 6180432.601)
            assert allclose(points[1][0],  222908.705)
            assert allclose(points[1][1], 6233785.284)
            
        self.failUnless(gsd.get_geo_reference().get_zone() == 56,
                        'Bad zone error!')
        points = gsd.get_data_points(as_lat_long=True)
        #print "test_lat_long_set points", points
        try:
            assert allclose(points[0][0], -34)
            assert allclose(points[0][1], 150)
        except AssertionError:
            assert allclose(points[1][0], -34)
            assert allclose(points[1][1], 150)

    def test_len(self):
        
        points = [[1.0, 2.1], [3.0, 5.3]]
        G = Geospatial_data(points)
        self.failUnless(2 ==len(G),  'Len error!')
        
        points = [[1.0, 2.1]]
        G = Geospatial_data(points)
        self.failUnless(1 ==len(G),  'Len error!')

        points = [[1.0, 2.1], [3.0, 5.3], [3.0, 5.3], [3.0, 5.3]]
        G = Geospatial_data(points)
        self.failUnless(4 ==len(G),  'Len error!')
        
    def test_split(self):
        """test if the results from spilt are disjoin sets"""
        
        points = [[1.0, 1.0], [1.0, 2.0],[1.0, 3.0], [1.0, 4.0], [1.0, 5.0],
                  [2.0, 1.0], [2.0, 2.0],[2.0, 3.0], [2.0, 4.0], [2.0, 5.0],
                  [3.0, 1.0], [3.0, 2.0],[3.0, 3.0], [3.0, 4.0], [3.0, 5.0],
                  [4.0, 1.0], [4.0, 2.0],[4.0, 3.0], [4.0, 4.0], [4.0, 5.0],
                  [5.0, 1.0], [5.0, 2.0],[5.0, 3.0], [5.0, 4.0], [5.0, 5.0]]
        attributes = {'depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                      'speed':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}
        G = Geospatial_data(points, attributes)
#        print G.get_data_points()
#        print G.get_attributes()

        factor = 0.21

        #will return G1 with 10% of points and G2 with 90%
        G1, G2  = G.split(factor) 
        
#        print 'len(G): %s len(G1): %s len(G2): %s' %(len(G), len(G1), len(G2))
#        print 'G: ', len(G),'G1: ', len(G1), 'G2: ', len(G2)

        assert allclose(len(G), len(G1)+len(G2))
        assert allclose(round(len(G)*factor), len(G1))
        
#        assert allclose(G == G1 + G2) must implentent __equal__
        
         
if __name__ == "__main__":

    #suite = unittest.makeSuite(Test_Geospatial_data, 'test_write_csv_attributes_lat_long')
    #suite = unittest.makeSuite(Test_Geospatial_data, 'verbose_test_load_pts_blocking')
    suite = unittest.makeSuite(Test_Geospatial_data, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)

    
