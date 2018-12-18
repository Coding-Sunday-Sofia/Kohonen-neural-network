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

import java.util.Random;

import janusz.rybarski.seweryn.habdank.activation.ActivationFunctionModel;

/**
 * Class representing <B>neuron</B> with bias and specyfied activation function
 * 
 * @author Janusz Rybarski e-mail: janusz.rybarski AT ae DOT krakow DOT pl
 * @author Seweryn Habdank-Wojew�dzki e-mail: habdank AT megapolis DOT pl
 * @version 1.0 2006/05/02
 * @see ActivationFunctionModel
 */

public class Neuron implements Cloneable {

	/**
	 * bias
	 */
	private double bias = 0;
	/**
	 * array of the weight
	 */
	private double[] weight;
	/**
	 * activation function
	 */
	private ActivationFunctionModel activationFunction;

	/**
	 * Creates new neuron with random weight and specyfied numbers of weight. Value
	 * of the weight is random where max value is definited by maxWeight.
	 * 
	 * @param weightNumber numbers of weight
	 * @param maxWeight    max value of the weight
	 */
	public Neuron(int weightNumber, double[] maxWeight) {
		if (weightNumber == maxWeight.length) {
			Random rand = new Random();
			weight = new double[weightNumber];
			for (int i = 0; i < weightNumber; i++) {
				weight[i] = rand.nextDouble() * maxWeight[i];
			}
		}
	}

	/**
	 * Creates new neuron with specyfied numbers of weight. Value of the weight is
	 * random where max value is definited by maxWeight. Activation function is
	 * definied by activationFunction parameter.
	 * 
	 * @param weightNumber       numbers of weight
	 * @param maxWeight          max value of the weight
	 * @param activationFunction activation function
	 */
	public Neuron(int weightNumber, double[] maxWeight, ActivationFunctionModel activationFunction) {
		this(weightNumber, maxWeight);
		this.activationFunction = activationFunction;
	}

	/**
	 * Creates new neuron with weight specyfied by array and specified activation
	 * funciton. Numbers of weights are the same as length of the array.
	 * 
	 * @param weightArray        array of the weight
	 * @param activationFunction activation function
	 */
	public Neuron(double[] weightArray, ActivationFunctionModel activationFunction) {
		int weightSize = weightArray.length;
		weight = new double[weightSize];
		for (int i = 0; i < weightSize; i++) {
			weight[i] = weightArray[i];
		}
		this.activationFunction = activationFunction;
	}

	/**
	 * Creates new neuron with weight specyfied by array. Numbers of weights are the
	 * same as length of the array.
	 * 
	 * @param weightArray array of the weight
	 */
	public Neuron(double[] weightArray) {
		int weightSize = weightArray.length;
		weight = new double[weightSize];
		for (int i = 0; i < weightSize; i++) {
			weight[i] = weightArray[i];
		}
	}

	/**
	 * Creates new neuron with weight specyfied by array, specified activation
	 * funciton and bias <br>
	 * Numbers of weights are the same as length of the array.
	 * 
	 * @param weightArray        array of the weight
	 * @param activationFunction activation function
	 * @param bias               bias
	 */
	public Neuron(double[] weightArray, ActivationFunctionModel activationFunction, double bias) {
		this(weightArray, activationFunction);
		this.bias = bias;
	}

	/**
	 * Return activation function or null if the activation function was not set.
	 * 
	 * @return activatin function
	 */
	public ActivationFunctionModel getActivationFunction() {
		return this.activationFunction;
	}

	/**
	 * Return value of the bias
	 * 
	 * @return value of the bias
	 */
	public double getBias() {
		return this.bias;
	}

	/**
	 * Returns array contains valu of the weights
	 * 
	 * @return array of the weights
	 */
	public double[] getWeight() {
		return this.weight;
	}

	/**
	 * Return value of the neuron after activation. if activation function is not
	 * set, function return sum of the multiplication of vector v_i and weight w_i
	 * for each i (numbers of weigth and lenght of the input vector)
	 * 
	 * @param inputVector input vector for neuron
	 * @return return
	 */
	public double getValue(double[] inputVector) {
		double value = 0;
		int inputSize = inputVector.length;
		for (int i = 0; i < inputSize; i++) {
			value = value + inputVector[i] * weight[i];
		}
		if (activationFunction != null)
			return activationFunction.getValue(value);
		else
			return value;
	}

	/**
	 * Set activation function
	 * 
	 * @param activationFunction reference to activation function
	 */
	public void setActivationFunction(ActivationFunctionModel activationFunction) {
		this.activationFunction = activationFunction;
	}

	/**
	 * Set bias to the neuron.
	 * 
	 * @param bias value of the bias
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}

	/**
	 * Set weigths from array as parameter
	 * 
	 * @param weight array of the weights
	 */
	public void setWeight(double[] weight) {
		this.weight = weight;
	}

	/**
	 * Return copy of teh object
	 * 
	 * @return copy of the object
	 */
	public Object clone() {
		try {
			Neuron neuronCopy = (Neuron) super.clone();
			neuronCopy.weight = (double[]) weight.clone();
			neuronCopy.activationFunction = activationFunction;
			neuronCopy.bias = bias;
			return neuronCopy;
		} catch (CloneNotSupportedException e) {
			return null;
		}
	}

	/**
	 * Returns a string representation of the Neuron. Neuron is describe by its
	 * weights [w_1,w_2,...]
	 * 
	 * @return a string representation of the Neuron
	 */
	public String toString() {
		String text = "";
		text += "[ ";
		int weightSize = weight.length;
		for (int i = 0; i < weightSize; i++) {
			text += weight[i];
			if (i < weightSize - 1) {
				text += ", ";
			}
		}
		text += " ]";
		return text;
	}
}
