/**
* Copyright (c) 2006, Seweryn Habdank-Wojewodzki
* Copyright (c) 2006, Janusz Rybarski
*
* All rights reserved.
* 
* Redistribution and use in source and binary forms,
* with or without modification, are permitted provided
* that the following conditions are met:
*
* Redistributions of source code must retain the above
* copyright notice, this list of conditions and the
* following disclaimer.
*
* Redistributions in binary form must reproduce the
* above copyright notice, this list of conditions
* and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
* AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
* WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
* THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
* USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
* WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package janusz.rybarski.seweryn.habdank.networks;

import janusz.rybarski.seweryn.habdank.kohenen.LearningData;
import janusz.rybarski.seweryn.habdank.kohenen.WTMLearningFunction;
import janusz.rybarski.seweryn.habdank.learning.ConstantFunctionalFactor;
import janusz.rybarski.seweryn.habdank.learning.LinearFunctionalFactor;
import janusz.rybarski.seweryn.habdank.kohenen.WTALearningFunction;
import janusz.rybarski.seweryn.habdank.topology.GaussNeighbourhoodFunction;
import janusz.rybarski.seweryn.habdank.metrics.EuclidesMetric;
import janusz.rybarski.seweryn.habdank.topology.MatrixTopology;
import janusz.rybarski.seweryn.habdank.kohenen.DefaultNetworkModel;

/**
 * Neural Network Class.
 * 
 * @author Janusz Rybarski e-mail: janusz.rybarski AT ae DOT krakow DOT pl
 * @author Seweryn Habdank-Wojewodzki e-mail: habdank AT megapolis DOT pl
 * @version 1.0 2006/05/02
 */

public class NeuralNetworks {

	/** Creates a new instance of NeuralNetworks */
	public NeuralNetworks() {
	}

	public static void main(String[] args) {
		int radius = 3; // neighbourhood size
		int maxIteration = 10; // max number of iteraiton

		System.out.println("Neural Network ver. 0.1 alpha");
		LearningData fileData = new LearningData("c:/trainning_data.txt"); // load sample data
		System.out.println("Generating new Network ....");
		MatrixTopology topology = new MatrixTopology(10, 10, radius); // creating new Matrics topology
		double[] maxWeight = { 200, 100 }; // weight interval from which random weigh are calculated

		// Create new network with random weight from definied interval and specified
		// topology
		DefaultNetworkModel network = new DefaultNetworkModel(2, maxWeight, topology); // generate network with 2
																						// weights (input for each
																						// neuron)

		System.out.println("Network was generated");
		ConstantFunctionalFactor constantFactor = new ConstantFunctionalFactor(0.8); // constatn learning factor
		GaussNeighbourhoodFunction neighboorhoodFunciton = new GaussNeighbourhoodFunction(radius); // Gause
																									// neighbourhood
																									// function

		/*
		 * Create WTA (Winer Takes All) learning algorythm for specified network, number
		 * of iteration, Euclides metrics funciton, specified learning data (fileData)
		 * and constatn learning factor
		 */

		WTALearningFunction learning = new WTALearningFunction(network, maxIteration, new EuclidesMetric(), fileData,
				constantFactor); // WTA learnig algorythm

		System.out.println("Learning .....");
		// learning.setShowComments(true); // show commentst diuring learning
		learning.learn(); // start learning
		System.out.println("Learning was finished");
		System.out.println(network); // print the neurons weights
		network.networkToFile("c:/network_after.txt"); // save weight after learning
	}
}
