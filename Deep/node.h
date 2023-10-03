#ifndef NODE_H
#define NODE_H
#include <svg.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <boost/preprocessor.hpp> // For enum to string.
#include <vector>
#include <iostream>
#include <memory>

// Macros for automatic convert enum to string, solution from stackoverflow. 
// https://stackoverflow.com/questions/5093460/how-to-convert-an-enum-type-variable-to-a-string
#define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)    \
    case elem : return BOOST_PP_STRINGIZE(elem);

#define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)                \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
    inline const char* ToString(name v)                                       \
    {                                                                         \
        switch (v)                                                            \
        {                                                                     \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }


using T = Eigen::MatrixXd;
using json = nlohmann::json;

/* Supply methods for json serialization and 
deserialization of Eigen Matrix. */
namespace nlohmann {
    template <>
    struct adl_serializer<T> {
        static void to_json(json& j, const T& opt);
        static void from_json(const json& j, T& opt);
    };
}

namespace Deep
{

DEFINE_ENUM_WITH_STRING_CONVERSIONS(gradFn, (none)(accumulateGrad)(transposeBackward)
    (matMulBackward)(reluBackward)(sumBackward)
    (addBackward)(addMmBackward)(subtractBackward)(mseBackward));

namespace svgUtility
{
/* Provided TopLeft coordinate of two Points (from->to),
dimension of nodes (nodeW, nodeH),
accurately create a polygon object that connects two Points. */
svg::Polygon getLink(svg::Point fromPoint, svg::Point toPoint, int nodeW, int nodeH);
};

class Node : public std::enable_shared_from_this<Node>
{
    public: 
        T data;
        T gradient;
        bool isLeaf;
        std::vector<std::shared_ptr<Node>> nextNodes;
        gradFn gradientFunction;
        /* Constructor. 
        Gradient must be zero-initialized, same shape as data.
        isLeaf should be used based on situation, 
        if isLeaf is true, gradFn will be accumulateGrad. 
        gradientFunction used to govern backward() behavior.
        */
        Node(T x, bool isleaf = true, 
            std::vector<std::shared_ptr<Node>> nextnodes = std::vector<std::shared_ptr<Node>>{}, 
            gradFn gradfn = gradFn::none);
        /* Overloading constructor for convenience */
        Node(T x, gradFn gradfn);
        /* Shape of the contained matrix (can be (M,N)) */
        std::vector<int> shape();
        /* Size, or number of elements in a matrix (can be M*N) */
        int size();

        /* Clear gradient */
        void zeroGrad();

        /* Overload transpose */
        std::shared_ptr<Node> transpose();
        /* ReLU */
        std::shared_ptr<Node> relu();
        /* Sum */
        std::shared_ptr<Node> sum();
        

        /* Backward */
        /* Backward for intermediate nodes. */
        void backward(T fromGradient);
        /* Backward for Loss, use a double, non-unit gradient. */
        void backward(double fromGradient);
        /* Backward for Loss, they typically uses default gradient of 1. */
        void backward();


        /* Overload << */
        friend std::ostream& operator<< (std::ostream &out, const std::shared_ptr<Node>& nodePtr);
        /* Show descendents (only their gradfn),
        return the total number of nodes */
        int descendents(int level = 0, bool verbose = false);
        int descendents(bool verbose);

        /* Visualize descendents */
        void visualizeGraph(std::string path);
};

}

#endif