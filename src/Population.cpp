#include "Population.h"
#include <time.h>
#include <stdlib.h>
#include <utility>
#include <cmath>
#include <set>
#include <iterator>

using namespace NEAT;
using namespace Neural;
using namespace std;

Population::Population(list<Actor*> *actors)
{
	//Set up the container
	this->nets = new list<NNetwork*>();
	this->actors = actors;

	//Set up the random number generator
	srand(time(NULL));

	//Start our innovation
	int inputs = actors->front()->getSensors();
	int outputs = actors->front()->getActions();
	weight_innovation = inputs * outputs;
	node_innovation = inputs + outputs;

	list<Actor*>::iterator it;
	for(it = actors->begin(); it != actors->end(); it++) {
		vector<NWeight> protoWeights;
		for(int i = 0; i < inputs; i++) {
			for(int j = 0; j < outputs; j++) {
				//Have our initial configuration be randomly weighted
				//connections between every input and output
				float weight = (((double) rand()) / RAND_MAX) * 2.0 - 1.0;
				protoWeights.push_back(NWeight(i, inputs + j, i * outputs + j, weight, true));
			}
		}
		vector<int> temp;
		NNetwork* net = new NNetwork(&protoWeights, inputs, outputs, &temp);
		nets->push_back(net);
		(*it)->setNetwork(net);
		net->setActor(*it);
	}
}

Population::~Population() 
{
	list<NNetwork*>::iterator it;
	for(it = nets->begin(); it != nets->end(); it++)
		delete(*it);

	delete nets;
}

void Population::stepGeneration()
{
	this->stepGeneration(false);
}

bool Population::compare_fitness(NNetwork* first, NNetwork* second)
{
	return first->getActor()->getFitness() < second->getActor()->getFitness();
}

void Population::stepGeneration(bool verbose)
{
	node_innovations.clear();
	new_weights_node_innovations.clear();
	weight_innovations.clear();

	//Time to see who lives and who dies
	//But first, some speciation!
	list<list<NNetwork*> > species;
	species.push_front(list<NNetwork*>());
	species.front().push_front(nets->front());

	list<NNetwork*> *newNetworks = new list<NNetwork*>();

	double fitnessSum = 0.0;
	double champion = 0.0;
	for(list<NNetwork*>::iterator it = nets->begin(); it != nets->end(); it++)
	{
		//Figure out which species it fits in
		bool found = false;
		for(list<list<NNetwork*> >::iterator it2 = species.begin();
					it2 != species.end() and not found; it2++) {

			if(NNetwork::calculateDelta(*it, (*it2).front(), c1, c3) <= speciationDifference){
				found = true;
				(*it2).push_back(*it);
			}
		}

		//If we didn't find one, start our own damn species
		if(not found) {
			species.push_front(list<NNetwork*>());
			species.front().push_front(*it);
		}
	}

	for(list<list<NNetwork*> >::iterator it = species.begin(); it != species.end(); it++) {
		for(list<NNetwork*>::iterator it2 = (*it).begin();
																		it2 != (*it).end();
																		it2++) {
			fitnessSum += (*it2)->getActor()->getFitness() / (*it).size();
			if((*it2)->getActor()->getFitness() > champion)
				champion = (*it2)->getActor()->getFitness();
		}
	}

	if(verbose) {
		printf("Total fitness %f\tChampion:\t%f\n", fitnessSum, champion);
		printf("Number of species this generation: %d\n", (int) species.size());
	}

	//Build the next generation		
	for(list<list<NNetwork*> >::iterator it = species.begin(); 
															it != species.end(); it++) {
		//Determine the species total fitness and champion
		float speciesFitness = 0.0f;
		float championFitness = 0.0f;
		NNetwork* champion;
		for(list<NNetwork*>::iterator it2 = (*it).begin();
																		it2 != (*it).end();
																		it2++) {
			speciesFitness += (*it2)->getActor()->getFitness() / (*it).size();
			if(championFitness < (*it2)->getActor()->getFitness()) {
				champion = (*it2);
				championFitness = (*it2)->getActor()->getFitness();
			}
		}

		//Use the species fitness to determine number of nets for next generation
		int nextGen = floor((speciesFitness / fitnessSum) * (double)actors->size());	
		if(verbose)
			printf("Nextgen: %d\t%f\n", nextGen, speciesFitness);

		//Now remove networks in the bottom half of the population
		//First sort by fitness
		(*it).sort(compare_fitness);
		//Remove the first half
		list<NNetwork*>::iterator half = (*it).begin();
		advance(half, (int) ((*it).size() * 0.75));
		(*it).erase((*it).begin(), half);

		//Now add in our designated number of networks by cloning current ones
		for(int i = 0; i < nextGen - 1; i++) {
			list<NNetwork*>::iterator randIter = (*it).begin();
			if((*it).size() > 1)
				advance(randIter, rand() % ((*it).size() - 1));
			newNetworks->push_back(mutateWeights(verbose, (*randIter)));
		}

		//Also add in our champion
		if(nextGen > 0)
			newNetworks->push_back(new NNetwork(champion->getWeights(),
																					champion->numInputs(),
																					champion->numOutputs(),
																					champion->getHidden()));
	}

	//Assign the new networks and delete the old ones
	for(list<NNetwork*>::iterator it = nets->begin(); it != nets->end(); it++) {
		delete (*it);
	}

	delete nets;

	nets = newNetworks;

	//Go through all our actors,
	//reset and give them new networks
	list<Actor*>::iterator it2;
	list<NNetwork*>::iterator it = nets->begin();
	for(it2 = actors->begin(); it2 != actors->end(); it2++) { (*it2)->clearNetwork(); }

	for(it2 = actors->begin(); it2 != actors->end() and it != nets->end(); it2++) {
		(*it2)->reset();
		(*it2)->setNetwork(*it);
		(*it)->setActor(*it2);
		it++;
	}
}

double Population::randRange(double lower, double upper)
{
	return (((double) rand()) / RAND_MAX) * (upper - lower) + lower;
}

NNetwork* Population::mutateWeights(bool verbose, NNetwork* net)
{

	//Grab the weights off of the network
	const vector<NWeight>* weights = net->getWeights();

	//Build a new list of weights
	vector<NWeight>* newWeights = new vector<NWeight>();

	//Also some new hidden nodes
	vector<int>* hidden = new vector<int>(*net->getHidden());

	//Do current weight mutations
	vector<NWeight>::const_iterator it2;
	for(it2 = weights->begin(); it2 != weights->end(); it2++) {
		//Determine if a mutation has occurred
		float weightVal = (*it2).getWeight();
		if(((double) rand()) / RAND_MAX <= weightMutationRate)
			weightVal += (((double) rand()) / RAND_MAX - 0.5) * 2 * weightMutationIntensity;

		//Also determine if a disabling has occurred
		bool enabled = (*it2).isEnabled();
		if(randRange(0, 1) <= weightDisableRate)
			enabled = false;

		newWeights->push_back(NWeight((*it2).getIn(),
																(*it2).getOut(),
																(*it2).getInnovation(),
																weightVal,
																enabled));
	}

	//Now determine if we're doing a mutation to add a weight 
	if(randRange(0, 1) <= addWeightMutationRate) {
		//If we're adding a weight, first determine unconnected nodes
		//O(N^2)
		bool found = false;
		int attempts = 0;
		while(not found and attempts < 5) {
			//Pick two nodes from our list of current nodes
			attempts++;
			int hiddenSize = hidden->size();

			//Our input node can be any node
			int nodeA = (int) randRange(0, hiddenSize + net->numInputs() + net->numOutputs() - 0.001);
			//If it's greater than the initial set of inputs and outputs,
			//do the correct mapping
			if(nodeA >= net->numInputs() + net->numOutputs()) {
				nodeA -= net->numInputs() + net->numOutputs();
				if(nodeA == hiddenSize)
					nodeA = hiddenSize - 1;
				nodeA = hidden->at(nodeA);
			}

			//Our outputs can only be outputs or hidden
			int nodeB = (int) randRange(0, hiddenSize + net->numOutputs() - 0.001);
			//If we're an output, be sure to do that
			if(nodeB < net->numOutputs()) {
				nodeB += net->numInputs();
			} else {
				nodeB -= net->numOutputs();
				if(nodeB == hiddenSize)
					nodeB = hiddenSize - 1;

				nodeB = net->getHidden()->at(nodeB);
			}

			//If it doesn't already exist
			if(not net->hasConnection(nodeA, nodeB)) {
				//if this innovation already happened, 
				//use the previous innovation number
				pair<int, int> weightPair(nodeA, nodeB);
				int innovation_number = weight_innovation;
				if(weight_innovations.count(weightPair))
					innovation_number = weight_innovations[weightPair];


				//build that weight!
				newWeights->push_back(NWeight(nodeA,
																			nodeB,
																			innovation_number,
																			randRange(-1.0, 1.0),
																			true));
				if(weight_innovations.count(weightPair) == 0) {
					weight_innovations[weightPair] = weight_innovation;
					weight_innovation++;
				}
			}
		}
	}

	//Now determine if we're doing a mutation to add a node
	if(randRange(0, 1) <= addNodeMutationRate) {
		//Pick a connection to split
		int weightNum = (int) randRange(0, net->numWeights());
		if(weightNum == (int) net->numWeights())
			weightNum = net->numWeights() - 1;

		const NWeight* weight = &(net->getWeights()->at(weightNum));
		weightNum = weight->getInnovation();

		//add the neurode
		//If it's already been added this generation, use that
		//innovation number
		int newNode = node_innovation;
		int weightInnoA, weightInnoB;
		if(node_innovations.count(weightNum) > 0) {
			newNode = node_innovations[weightNum];
			weightInnoA = new_weights_node_innovations[weightNum].first;
			weightInnoB = new_weights_node_innovations[weightNum].second;
		} else {
			node_innovations[weightNum] = newNode;
			node_innovation++;

			weightInnoA = weight_innovation;
			weight_innovation++;
			weightInnoB = weight_innovation;
			weight_innovation++;
			new_weights_node_innovations[weightNum] 
				= pair<int, int>(weightInnoA, weightInnoB);
		}

		//Add in the hidden node
		hidden->push_back(newNode);

		//Add in our two weights
		//old node A to new node
		newWeights->push_back(NWeight(weight->getIn(),
																	newNode,
																	weightInnoA,
																	randRange(-1.0, 1.0),
																	true));
		//new node to old node B
		newWeights->push_back(NWeight(newNode,
																	weight->getOut(),
																	weightInnoB,
																	randRange(-1.0, 1.0),
																	true));
		//Also disable the old weight
		//Find it and disable it
		vector<NWeight>::iterator itN;
		for(itN = newWeights->begin(); itN != newWeights->end(); itN++) {
			if((*itN).getInnovation() == weightNum)
				(*itN).setEnabled(false);
		}
	}

	//Build a new network with the mutated weights and nodes
	NNetwork* newNet = new NNetwork(newWeights,
																	net->numInputs(),
																	net->numOutputs(),
																	hidden);
	delete newWeights;
	delete hidden;
	return newNet;
}

void Population::printGeneration()
{
	list<NNetwork*>::iterator it;
	for(it = nets->begin(); it != nets->end(); it++) {
		(*it)->printNetwork();
	}
}
