#include "NWeight.h"

using namespace Neural;

NWeight::NWeight(int in, int out, int innovation, float weight, bool enabled)
{
	this->in = in;
	this->out = out;
	this->innovation = innovation;
	this->weight = weight;
	this->enabled = enabled;
}
