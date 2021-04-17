#!/usr/bin/python

#from __future__ import absolute_import, division, with_statement

__all__ = ['berry_phase']
__author__  = ['Jorge I. Facio']
__date__    = 'April 7, 2021'
__email__   = 'facio.ji@gmail.com'
__version__ = '0.'


import sys,os,re,string,shutil,time
from subprocess import call
import numpy as np
import numpy.linalg as LA
import math
import h5py
import pyfplo.slabify as sla

print( '\npyfplo version=: {0}\nfrom: {1}\n'.format(sla.version,sla.__file__))
# protect against wrong version
#if fedit.version!='21.00': raise RuntimeError('pyfplo version is incorrect.')

class chain:
  """
  The goal is to compute the Berry phase along a closed chain in momentum space. The closure is enforced by the periodicity of Bloch states. 

  The chain is determined by a lattice vector G and by the number of subdivisions N 

  """

  def link(self,ik_1,ik_2,correct_wc=False):
      """
      Here we implement the link determinant between points k1 and k2 (det in Eq. 50 of RMP by Resta)

      Args:

         ik_1: index of first k-point

         ik_2: index of second k-point

         correct_wc: Boolean only used when computing the periodic link in the relative gauge.
      """
      print(ik_1,ik_2)
      Norb = len(self.eig[0][0])
      one = np.identity(Norb)

      Ck1_dag = np.conj(self.eig[ik_1][1].T)
      Ck2 = self.eig[ik_2][1]

      #we compute to the projection of Berry connection along the G vector defining the chain.
      G_unit = self.G / LA.norm(self.G)

      A_G = G_unit[0] * self.eig[ik_1][2][0] + G_unit[1] * self.eig[ik_1][2][1] + G_unit[2] * self.eig[ik_1][2][2] 
      #to do: consider different schemes for computing A_G. Here we are using the one that uses k_1, we should test a centered scheme as well.

      if not correct_wc:

          M = LA.multi_dot([ Ck1_dag, one - 1j * self.h * A_G, Ck2])

      else:

          #WCexp is an auxiliary diagonal matrix that contains e^{-iG dot \vec{s}} in its s-th entry, where \vec{s} is the wannier center 
          WC = self.S.wannierCenterMatrix()
 #         print(WC)
          WCexp = np.identity(Norb)
          for i_x in range(3):
              exp = np.identity(Norb)
              for i_orb in range(Norb):
#                  print(np.exp(-1j * self.G[i_x]*WC[i_x][i_orb,i_orb]))
                  exp[i_orb,i_orb]  = np.exp(-1j * self.G[i_x]*WC[i_x][i_orb,i_orb])
              WCexp = LA.multi_dot([WCexp,exp])

          M = LA.multi_dot([ Ck1_dag, one - 1j * self.h * A_G, WCexp ,Ck2])

      return LA.det(M[0:self.Nbands,0:self.Nbands])

  def periodic_link(self):
      """
      Here we implement the periodic link.

      We consider the two possibilites of Bloch gauge.
      """

      if(self.gauge=='periodic'):
          return self.link(self.N-2,0)
      if(self.gauge=='relative'):
          return self.link(self.N-2,0,correct_wc=True)

  def berry_phase(self):
      """
      Here we compute the Berry phase associated with the chain.
      """
      det = 1
      for i_k in range(self.N-2):

          det *= self.link(i_k,i_k+1)

      return det * self.periodic_link()

  def eigen_all(self):
      """
      Here we diagonalize Hk along the chain.

      Approach: we store in memory all eigenvalues and eigenvector info of all the "different" k-points in the chain that we will use later to compute the links. Notice we exclude the Nth point which is equal to the first one by periodicity.
      
      """
      self.eig = []
      for i_k in range(self.N-1):
          k = self.kpoints[i_k]

          (Hk,dHk,Abk) = self.S.hamAtKPoint(k * self.S.kscale,self.ms,gauge=self.gauge,makedhk=True,makebasisconnection=True)
          (E,C)=self.S.diagonalize(Hk)
          BasisConnection = [Abk[0],Abk[1],Abk[2]]
          self.eig.append([E,C,BasisConnection])

  def __init__(self,**kwargs):
      """
      Args:
 
      origin: k-point at which the chain starts.

      G: lattice vector along which the chain is oriented.

      Nbands: number of occupied bands.

      N: number of points.
      """

      self.S = kwargs['slabify']
      self.N = kwargs['N']
      self.Nbands = kwargs['Nbands']
      self.G = kwargs['G']
      self.origin = kwargs['origin']
      self.verbosity = kwargs.pop('verbosity',0)
      self.gauge = kwargs.pop('gauge','periodic')
      self.ms  = 0 #for the moment we always have spin-orbit coupling

      #kpoints contains N points. The first one and the last one are the same by periodicity and will be treated differently
      self.kpoints = [self.origin+self.G * i /self.N for i in range(self.N)]
      self.h = LA.norm(self.G)/(self.N-1)

      print("Using gauge, ", self.gauge)
      print("Chaing oriented along:  ", self.G)
      print("Number of points:  ", self.N)
      print("k-scale: ", self.S.kscale)

      print("kpoins: ", self.kpoints)
      self.eigen_all()
