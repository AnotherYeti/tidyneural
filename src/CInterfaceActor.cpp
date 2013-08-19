#include "CInterfaceActor.h"
#include "Population.h"

using namespace Neural;
using namespace NEAT;

CActor::CActor(float (fitnessFunc()), int sensors, int actions, void resetFunc(), void destroyFunc())
{
	this->sensors = sensors;
	this->actions = actions;
	this->fitnessFunc = fitnessFunc;
	this->destroyFunc = destroyFunc;
	this->resetFunc = resetFunc;
}

CActor::~CActor()
{
	destroyFunc();
}

float CActor::getFitness()
{
	return fitnessFunc();
}

void CActor::reset()
{
	resetFunc();
}

//All the c interface functions
void* buildActor(float fitnessFunc(), int sensors, int actions, void resetFunc(), void destroyFunc())
{
	return new CActor(fitnessFunc, sensors, actions, resetFunc, destroyFunc);
}

void setInput(void* actor, int input, float value)
{
	((CActor*) actor)->getNetwork()->setInput(input, value);
}

float getOutput(void* actor, int output)
{
	return ((CActor*) actor)->getNetwork()->getOutput(output);
}

bool hasNetwork(void* actor)
{
	return ((CActor*) actor)->hasNetwork();
}

void stepNetwork(void* actor)
{
	((CActor*) actor)->getNetwork()->step();
}

void* buildPopulation(void** actorsIn, int size)
{
	std::list<Actor*> *actors = new std::list<Actor*>();
	for(int i = 0; i < size; i++) {
		actors->push_back((Actor*) actorsIn[i]);
	}

	Population* pop = new Population(actors);
	return pop;
}

void setWeightMutationRate(void* population, float newRate)
{
	((Population*)population)->weightMutationRate = newRate;
}

void setWeightMutationIntensity(void* population, float newRate)
{
	((Population*)population)->weightMutationIntensity = newRate;
}

void setWeightDisableRate(void* population, float newRate)
{
	((Population*)population)->weightDisableRate = newRate;
}

void setAddWeightRate(void* population, float newRate)
{
	((Population*)population)->addWeightMutationRate = newRate;
}

void setAddNodeRate(void* population, float newRate)
{
	((Population*)population)->addNodeMutationRate = newRate;
}
void setc1(void* population, float c1) 
{
	((Population*)population)->c1 = c1;
}

void setc3(void* population, float c3)
{
	((Population*)population)->c3 = c3;
}
void setSpeciationDifference(void* population, float newSpeciation)
{
	((Population*)population)->speciationDifference = newSpeciation;
}

void stepPopulationGeneration(void* population)
{
	((Population*)population)->stepGeneration();
}

void destroyPopulation(void* population)
{
	std::list<Actor*>* actors = ((Population*)population)->getActors();
	for(std::list<Actor*>::iterator it = actors->begin(); it != actors->end(); it++) {
		delete *it;
	}

	delete actors;

	delete (Population*)population;
}
