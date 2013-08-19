#include "NNetwork.h"
#include "math.h"
#include <stdio.h>

using namespace Neural;
using namespace std;

NNetwork::NNetwork(const vector<NWeight> *weightsIn, int inputs, int outputs, const vector<int> *hidden)
{
	//Initialize our private stuff
	this->weights = new vector<NWeight>(*weightsIn);
	this->hiddenNodes = new vector<int>(*hidden);
	this->inputs = inputs;
	this->outputs = outputs;
	this->neurodes = new map<int, float>();
	this->neurodeBuffer = new map<int, float>();
	this->numweights = (int) weights->size();

	//And initialize the proper neurodes
	for(int i = 0; i < inputs + outputs; i++) {
		(*neurodes)[i] = 0.0f;
		(*neurodeBuffer)[i] = (*neurodes)[i];
	}

	vector<int>::iterator it;
	for(it = hiddenNodes->begin(); it != hiddenNodes->end(); it++) {
		(*neurodes)[*it] = 0.0f;
		(*neurodeBuffer)[*it] = (*neurodes)[*it];
	}
}

NNetwork::NNetwork(const NNetwork* n)
{
	//Do the copy
	NNetwork(n->getWeights(), n->numInputs(), n->numOutputs(), n->getHidden());
}

NNetwork::~NNetwork()
{
	delete weights;
	delete hiddenNodes;
	delete neurodes;
	delete neurodeBuffer;
}

float NNetwork::getOutput(int output) 
{
	if(output >= 0 && output < outputs)
		return (*neurodes)[inputs + output];
	else
		return 0.0f;
}

void NNetwork::setInput(int input, float value) 
{
	if(input >= 0 && input < inputs) {
		(*neurodes)[input] = value;
		(*neurodeBuffer)[input] = (*neurodes)[input];
	}
}

void NNetwork::step() 
{
	//Make sure the buffer is a copy
	map<int, float>::iterator it;
	for(it = neurodes->begin(); it != neurodes->end(); it++) {
		(*neurodeBuffer)[it->first] = it->second;
	}
	
	//Go through all our weights and set the buffer
	vector<NWeight>::iterator it2;
	for(it2 = weights->begin(); it2 != weights->end(); it2++)
	{
		NWeight weight = *it2;
		if(weight.isEnabled()) {
			(*neurodeBuffer)[weight.getOut()] += weight.getWeight() * (*neurodes)[weight.getIn()];
		}
	}

	//Apply the sigmoid
	for(it = neurodeBuffer->begin(); it != neurodeBuffer->end(); it++) {
		(*it).second = tanh(it->second);
	}

	//Flip the buffer
	map<int, float>* tmp = neurodes;
	neurodes = neurodeBuffer;
	neurodeBuffer = tmp;
}

void NNetwork::reset()
{
	map<int, float>::iterator it;
	for(it = neurodes->begin(); it != neurodes->end(); it++) {
		(it->second) = 0.0f;
	}
}

bool NNetwork::hasConnection(int nodeA, int nodeB)
{
	//Iterate through our weights and look for the connection
	bool found = false;
	vector<NWeight>::iterator it;
	for(it = weights->begin(); it != weights->end() and not found; it++) {
		if((*it).getIn() == nodeA and (*it).getOut() == nodeB) 
			found = true;
	}

	return found;
}

void NNetwork::printNetwork() 
{
	printf("Net:\n");
	printf("Weight Genome:\n");
	printf("In\tOut\tWeight\tInnovation\tEnabled\n");
	vector<NWeight>::iterator it;
	for(it = weights->begin(); it != weights->end(); it++) {
		printf("%d\t%d\t%.3g\t%d\t\t%s\n", (*it).getIn(),
																(*it).getOut(),
																(*it).getWeight(),
																(*it).getInnovation(),
																(*it).isEnabled() ? "O" : "·");
	}
}

//See NEAT paper, page 110
float NNetwork::calculateDelta(const NNetwork* a, const NNetwork* b, float c1, float c3)
{
	//Simple, slow, N^2 comparison
	int eAndD = 0;
	int overlap = 0;
	double weightDiffSum = 0.0;

	//Weights a has that b doesn't
	vector<NWeight>::const_iterator it1, it2;
	for(it1 = a->getWeights()->begin(); it1 != a->getWeights()->end(); it1++) {
		//Current weight
		int weightInno = (*it1).getInnovation();
		bool found = false;

		for(it2 = b->getWeights()->begin(); it2 != b->getWeights()->end() and not found; it2++) {

			//If we found a value, be sure to update our weight differences
			if((*it2).getInnovation() == weightInno) {
				found = true;
				overlap++;
				weightDiffSum += fabs((*it2).getWeight() - (*it1).getWeight());
			}
		}

		if(not found) eAndD++;
	}

	//Weights b has a doesn't
	for(it1 = b->getWeights()->begin(); it1 != b->getWeights()->end(); it1++) {
		//Current weight
		int weightInno = (*it1).getInnovation();
		bool found = false;
		for(it2 = a->getWeights()->begin(); it2 != a->getWeights()->end() and not found; it2++) {
			if((*it2).getInnovation() == weightInno) {
				found = true;
			}
		}

		if(not found) eAndD++;
	}

	//Figure out which network has more
	double maxWeights = a->numWeights() > b->numWeights() 
												? a->numWeights()
												: b->numWeights();

	//return the d
	return (((double) eAndD) * c1) / maxWeights + c3 * weightDiffSum / overlap;
}
