/**
�* Copyright (c) 2006, Seweryn Habdank-Wojewodzki
�* Copyright (c) 2006, Janusz Rybarski
�*
�* All rights reserved.
�* 
�* Redistribution and use in source and binary forms,
�* with or without modification, are permitted provided
�* that the following conditions are met:
�*
�* Redistributions of source code must retain the above
�* copyright notice, this list of conditions and the
�* following disclaimer.
�*
�* Redistributions in binary form must reproduce the
�* above copyright notice, this list of conditions
�* and the following disclaimer in the documentation
�* and/or other materials provided with the distribution.
�*
�* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
�* AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
�* WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
�* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
�* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
�* THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
�* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
�* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
�* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
�* USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
�* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
�* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
�* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
�* WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
�* OF THE POSSIBILITY OF SUCH DAMAGE.
�*/

package janusz.rybarski.seweryn.habdank.kohenen;

import janusz.rybarski.seweryn.habdank.metrics.MetricModel;
import java.util.ArrayList;

import janusz.rybarski.seweryn.habdank.learning.LearningFactorFunctionalModel;

/**
 * <I>Winner Takes All</I> - algorytm there only winnig neuron weights are
 * changed according to the formula w(k+1) = w(k) + n * (x-w) where <br>
 * <I>w(k+1)</I> - neuron weight in <I>k +1</I> interation <br>
 * <I>w(k)</I> - neruon weight for <I>k</I> iteration <br>
 * <I>n</I> - value of learning function factor for <I> k </I> iteriation<br>
 * <I>x</I> - learning vector od data <I>w</I> - neuron weight
 * 
 * @author Janusz Rybarski e-mail: janusz.rybarski AT ae DOT krakow DOT pl
 * @author Seweryn Habdank-Wojew�dzki e-mail: habdank AT megapolis DOT pl
 * @version 1.0 2006/05/02
 * @see WTMLearningFunction
 */
public class WTALearningFunction {

	private MetricModel metrics;
	private NetworkModel networkModel;
	private int maxIteration;
	private LearningDataModel learningData;
	private LearningFactorFunctionalModel functionalModel;
	private boolean showComments;

	/**
	 * Creates a new instance of WTALearningFunction.
	 * 
	 * @param networkModel    network model
	 * @param maxIteration    iteration number
	 * @param metrics         metrics
	 * @param learningData    learnig data
	 * @param functionalModel functional model
	 * @see MetricModel
	 * @see LearningData
	 * @see NetworkModel
	 * @see LearningFactorFunctionalModel
	 */
	public WTALearningFunction(NetworkModel networkModel, int maxIteration, MetricModel metrics,
			LearningDataModel learningData, LearningFactorFunctionalModel functionalModel) {
		this.maxIteration = maxIteration;
		this.networkModel = networkModel;
		this.metrics = metrics;
		this.learningData = learningData;
		this.functionalModel = functionalModel;
	}

	/**
	 * Return information if learning process dispalys information about learning
	 * process.
	 * 
	 * @return true if learning process display informatio
	 */
	public boolean isShowComments() {
		return showComments;
	}

	/**
	 * Set if comments durring learning process must be shown.
	 * 
	 * @param showComments <B>true</B> if comments must be shown, <B>false</B>
	 *                     otherwise
	 */
	public void setShowComments(boolean showComments) {
		this.showComments = showComments;
	}

	/**
	 * Rerutn metrics
	 * 
	 * @return metrics
	 * @see MetricModel
	 */
	public MetricModel getMetrics() {
		return metrics;
	}

	/**
	 * Set metrics
	 * 
	 * @param metrics metrics
	 */
	public void setMetrics(MetricModel metrics) {
		this.metrics = metrics;
	}

	public void setNetworkModel(NetworkModel networkModel) {
		this.networkModel = networkModel;
	}

	/**
	 * Return network model
	 * 
	 * @return network model
	 */
	public NetworkModel getNetworkModel() {
		return networkModel;
	}

	/**
	 * Set max interation
	 * 
	 * @param maxIteration max interation
	 */
	public void setMaxIteration(int maxIteration) {
		this.maxIteration = maxIteration;
	}

	public int getMaxIteration() {
		return maxIteration;
	}

	public void setLearningData(LearningDataModel learningData) {
		this.learningData = learningData;
	}

	public LearningDataModel getLearningData() {
		return learningData;
	}

	public void setFunctionalModel(LearningFactorFunctionalModel functionalModel) {
		this.functionalModel = functionalModel;
	}

	public LearningFactorFunctionalModel getFunctionalModel() {
		return functionalModel;
	}

	private int getBestNeuron(double[] vector) {
		Neuron tempNeuron;
		double distance, bestDistance;
		int networkSize = networkModel.getNumbersOfNeurons();
		int bestNeuron = 0;
		tempNeuron = networkModel.getNeuron(0);
		bestDistance = metrics.getDistance(tempNeuron.getWeight(), vector);
		for (int i = 1; i < networkSize; i++) {
			tempNeuron = networkModel.getNeuron(i);
			distance = metrics.getDistance(tempNeuron.getWeight(), vector);
			if (distance < bestDistance) {
				bestDistance = distance;
				bestNeuron = i;
			}
		}
		return bestNeuron;
	}

	private void changeNeuronWeight(int neuronNumber, double[] vector, int iteration) {
		double[] weightList = networkModel.getNeuron(neuronNumber).getWeight();
		int weightNumber = weightList.length;
		double weight;
		if (showComments) {
			String vectorText = "[";
			for (int i = 0; i < vector.length; i++) {
				vectorText += vector[i];
				if (i < vector.length - 1) {
					vectorText += ", ";
				}
			}
			vectorText += "]";
			System.out.println("Vector: " + vectorText);
			String weightText = "[";
			for (int i = 0; i < weightList.length; i++) {
				weightText += weightList[i];
				if (i < weightList.length - 1) {
					weightText += ", ";
				}
			}
			weightText += "]";
			System.out.println("Neuron " + (neuronNumber + 1) + " weight before change: " + weightText);
		}
		for (int i = 0; i < weightNumber; i++) {
			weight = weightList[i];
			weightList[i] += functionalModel.getValue(iteration) * (vector[i] - weight);
		}
		if (showComments) {
			String weightText = "[";
			for (int i = 0; i < weightList.length; i++) {
				weightText += weightList[i];
				if (i < weightList.length - 1) {
					weightText += ", ";
				}
			}
			weightText += "]";
			System.out.println("Neuron " + (neuronNumber + 1) + " weight after change: " + weightText);
		}
	}

	/**
	 * Start learning process
	 */
	public void learn() {
		int bestNeuron = 0;
		double[] vector;

		int dataSize = learningData.getDataSize();
		for (int i = 0; i < maxIteration; i++) {
			if (showComments) {
				System.out.println("Iteration number: " + (i + 1));
			}
			for (int j = 0; j < dataSize; j++) {
				vector = learningData.getData(j);
				bestNeuron = getBestNeuron(vector);
				if (showComments) {
					System.out.println("Best neuron number: " + (bestNeuron + 1));
				}
				changeNeuronWeight(bestNeuron, vector, i);
			}
		}
	}
}
