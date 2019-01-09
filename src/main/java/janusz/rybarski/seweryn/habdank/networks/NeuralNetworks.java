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
import janusz.rybarski.seweryn.habdank.learning.GaussFunctionalFactor;
import janusz.rybarski.seweryn.habdank.learning.LearningFactorFunctionalModel;
import janusz.rybarski.seweryn.habdank.learning.LinearFunctionalFactor;
import janusz.rybarski.seweryn.habdank.kohenen.WTALearningFunction;
import janusz.rybarski.seweryn.habdank.topology.GaussNeighbourhoodFunction;
import janusz.rybarski.seweryn.habdank.metrics.EuclidesMetric;
import janusz.rybarski.seweryn.habdank.topology.MatrixTopology;
import janusz.rybarski.seweryn.habdank.topology.NeighbourhoodFunctionModel;
import janusz.rybarski.seweryn.habdank.kohenen.DefaultNetworkModel;

/**
 * Neural Network Class.
 * 
 * @author Janusz Rybarski e-mail: janusz.rybarski AT ae DOT krakow DOT pl
 * @author Seweryn Habdank-Wojewodzki e-mail: habdank AT megapolis DOT pl
 * 
 * @version 1.0 2006/05/02
 */
public class NeuralNetworks {

	/** Creates a new instance of the class. */
	private NeuralNetworks() {
	}

	public static void main(String[] args) {
		/* Neighborhood size. */
		int radius = 3;

		/* Max number of iteration. */
		int iterations = 10;

		System.out.println("Neural Network v.0.0.1 (alpha).");

		/* Load sample data. */
		LearningData data = new LearningData("./dat/trainning_data.txt");

		System.out.println("Generating new network ...");

		/* Creating new matrices topology. */
		MatrixTopology topology = new MatrixTopology(10, 10, radius);

		/* Weight interval from which random weigh are calculated. */
		double[] interval = { 200, 100 };

		/*
		 * Create new network with random weight from defined interval and specified
		 * topology. Generate network has 2 weights (input for each neuron).
		 */
		DefaultNetworkModel network = new DefaultNetworkModel(interval.length, interval, topology);

		System.out.println("Network was generated ...");

		/* Constant learning factor. */
		LearningFactorFunctionalModel factor = new GaussFunctionalFactor(0.8);

		/*
		 * Create WTA (Winer Takes All) learning algorithm for specified network, number
		 * of iteration, Euclides metric metrics function, specified learning data
		 * (fileData) and constant learning factor.
		 */
		WTALearningFunction learning = new WTALearningFunction(network, iterations, new EuclidesMetric(), data, factor);

		System.out.println("Learning ...");

		/* Show comments during learning. */
//		 learning.setShowComments(true);
		learning.learn();

		System.out.println("Learning was finished ...");

		/* Print the neurons weights. */
		System.out.println(network);

		/* Save weight after learning. */
		network.networkToFile("./dat/network_after.txt");
	}
}
