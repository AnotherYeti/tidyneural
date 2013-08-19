#include "Actor.h"

using namespace Neural;

Actor::Actor() {
	this->network = 0;
}

Actor::~Actor() {}

void Actor::setNetwork(NNetwork* newNet)
{
	this->network = newNet;
}

void Actor::clearNetwork() 
{
	this->network = 0;
}

bool Actor::hasNetwork()
{
	return network != 0;
}
