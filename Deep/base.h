#ifndef BASE_H
#define BASE_H

namespace Deep
{
/* All layer types should inherited from Layer class. 
Layer is the base class of all layers that have parameters.
Ex: Fully Connected Layer should be derived from Layer,
but MaxPooling should not be. */
class Layer {};
}

#endif