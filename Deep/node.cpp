#include "node.h"
#include <svg.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <set>
#include <unordered_map>
#include <cmath>

using T = Eigen::MatrixXd;
using NSP = std::shared_ptr<Deep::Node>;
using namespace svg;

namespace nlohmann {
    void adl_serializer<T>::to_json(json& j, const T& data)
    {
        T mat { data }; // Force data to be column major;
        std::vector<double> vecForm(mat.data(), mat.data() + mat.rows() * mat.cols());
        j = json{{"row", data.rows()}, {"array", vecForm}};
    }
    void adl_serializer<T>::from_json(const json& j, T& data)
    {
        // Will convert to column major matrix.
        int row {j["row"].template get<int>()};
        std::vector<double> numbers {j["array"].template get<std::vector<double>>()};
        int N {static_cast<int>(numbers.size())};
        int col { N / row};
        Eigen::Map<Eigen::MatrixXd> newData(numbers.data(), row, col);
        data = newData;
    }
}

namespace Deep
{

svg::Polygon svgUtility::getLink(svg::Point fromPoint, svg::Point toPoint, int nodeW, int nodeH)
{
    constexpr double sqrt2by2 { 0.707107 };
    svg::Polygon arrow(Stroke(.5, Color::Blue));
    int wingLen {nodeH / 2}; // Length of arrow wings
    /* Guard no operation */
    if (fromPoint.x == toPoint.x && fromPoint.y == toPoint.y)
        return arrow;
    svg::Point exactFrom {};
    svg::Point exactTo {};
    if (fromPoint.y < toPoint.y)
    {
        exactFrom = svg::Point(fromPoint.x + nodeW / 2, fromPoint.y + nodeH);
        exactTo = svg::Point(toPoint.x + nodeW / 2, toPoint.y);
    }
    else if (fromPoint.y == toPoint.y && fromPoint.x < toPoint.x)
    {
        exactFrom = svg::Point(fromPoint.x + nodeW, fromPoint.y + nodeH / 2);
        exactTo = svg::Point(toPoint.x, toPoint.y + nodeH / 2);
    }
    else if (fromPoint.y == toPoint.y && fromPoint.x > toPoint.x)
    {
        exactFrom = svg::Point(fromPoint.x, fromPoint.y + nodeH / 2);
        exactTo = svg::Point(toPoint.x + nodeW, toPoint.y + nodeH / 2);
    }
    else 
    {
        exactFrom = svg::Point(fromPoint.x + nodeW / 2, fromPoint.y);
        exactTo = svg::Point(toPoint.x + nodeW / 2, toPoint.y + nodeH);
    }
    svg::Point vec(exactTo.x - exactFrom.x, exactTo.y - exactFrom.y);
    double arrowLength { std::sqrt(vec.x * vec.x + vec.y * vec.y) };
    
    vec = svg::Point(vec.x * wingLen / arrowLength, vec.y * wingLen / arrowLength);
    svg::Point leftWing(
        static_cast<int>(exactTo.x + sqrt2by2 * (-vec.x +vec.y)), 
        static_cast<int>(exactTo.y - sqrt2by2 * (vec.x+vec.y))
    );
    svg::Point rightWing(
        static_cast<int>(exactTo.x - sqrt2by2 * (vec.x+vec.y)), 
        static_cast<int>(exactTo.y + sqrt2by2 * (vec.x-vec.y))
    );

    arrow << exactFrom << exactTo << leftWing << exactTo << rightWing << exactTo << exactFrom;
    return arrow;
}

std::ostream& operator<<(std::ostream& out, const gradFn gf){
    out << ToString(gf);  
    return out;
}

Node::Node(T x, bool isleaf, std::vector<std::shared_ptr<Node>> nextnodes, gradFn gradfn):
    data(x), gradient(T{}), isLeaf(isleaf), nextNodes(nextnodes), gradientFunction(gradfn)
{
    // zero-initialize gradient
    gradient = T::Zero(x.rows(), x.cols());
    // if isleaf is true, verify gradfn is accumulateGrad. 
    // else throw and error.
    if (isleaf && (gradientFunction != Deep::gradFn::accumulateGrad) 
        && (gradientFunction != Deep::gradFn::none))
    {
        throw std::invalid_argument(
            "For a leaf node, gradient function must be Deep::gradFn::accumulateGrad or none."
        );
    }
    // if isleaf is true, there shouldn't be any next nodes.
    if (isleaf && (nextNodes.size() != 0))
    {
        throw std::invalid_argument(
            "A leaf node shouldn't have next nodes."
        );
    }  
};

Node::Node(T x, gradFn gradfn): 
    Node(x, true, std::vector<std::shared_ptr<Node>>{}, gradfn) {}

std::vector<int> Node::shape()
{
    return std::vector<int> { static_cast<int>(data.rows()), static_cast<int>(data.cols()) };
}

int Node::size()
{
    return static_cast<int>(data.size());
}

void Node::zeroGrad()
{
    gradient.fill(0.0);
}

/* Overloading operators */
std::shared_ptr<Node> Node::transpose()
{
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            this -> data.transpose(),
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::transposeBackward
        )
    );
    
    return nodePtr;
}

/* Class ReLU */
std::shared_ptr<Node> Node::relu()
{
    Eigen::MatrixXd x { this->data };
    x = x.unaryExpr([](double num){
        return (num>0) ? num : 0.0;
    });
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            x,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::reluBackward
        )
    );
    return nodePtr;
}


std::shared_ptr<Node> Node::sum()
{
    Eigen::MatrixXd summation(1,1);
    summation << this -> data.sum();
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            summation,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::sumBackward
        )
    );
    return nodePtr;
}


void Node::backward(T fromGradient)
{
    /* Refuse if gradfn is none. */
    if (this->gradientFunction == gradFn::none)
        return ;
    /* gradient should have the same shape as data */
    assert((this->data.rows() == fromGradient.rows()) 
        && "Gradient doesn't have the same number of rows as data.");
    assert((this->data.cols() == fromGradient.cols()) 
        && "Gradient doesn't have the same number of cols as data.");
    
    // First solve accumulate Grad, which is a leaf node.
    if (this->gradientFunction == gradFn::accumulateGrad)
    {
        gradient += fromGradient;
        return;
    }
    // For other, get the shape of first next node.
    switch (this->gradientFunction)
    { 
        /* temporary toGradient destroyed after switch, so it is not stored.
        I should think of a way for user to retain gradients. TODO */ 
        case gradFn::transposeBackward:
        {
            this->nextNodes[0]->backward(fromGradient.transpose());
            break;
        }
        case gradFn::matMulBackward:
        {
            T data1 { this->nextNodes[0]->data };
            T data2 { this->nextNodes[1]->data };
            this->nextNodes[0]->backward(fromGradient * data2.transpose());
            this->nextNodes[1]->backward(data1.transpose() * fromGradient);
            break;
        }
        case gradFn::reluBackward:
        {
            T mask { this->nextNodes[0]->data };
            mask = mask.unaryExpr([](double num){
                return static_cast<double>(num>0);
            });
            this->nextNodes[0]->backward(fromGradient.cwiseProduct(mask));
            break;
        }
        case gradFn::sumBackward:
        {
            T toGradient{ this->nextNodes[0]->data };
            /* sumBackward node must contains a scalar [1,1] matrix. */
            toGradient.fill(fromGradient(0,0));
            this->nextNodes[0]->backward(toGradient);
            break;
        } 
        case gradFn::addBackward:
        {
            /* Simple propagate the gradient to the 
            next two components. */
            this->nextNodes[0]->backward(fromGradient);
            this->nextNodes[1]->backward(fromGradient);
            break;
        }
        case gradFn::addMmBackward:
        {
            this->nextNodes[0]->backward(fromGradient.colwise().sum().transpose());
            this->nextNodes[1]->backward(fromGradient * this->nextNodes[2]->data.transpose());
            this->nextNodes[2]->backward(this->nextNodes[1]->data.transpose() * fromGradient);
            break;
        }
        case gradFn::subtractBackward:
        {
            this->nextNodes[0]->backward(fromGradient);
            this->nextNodes[1]->backward(-fromGradient);
            break;
        }
        case gradFn::mseBackward:
        {
            const int N { this->nextNodes[0]->size() };
            /* Gradient for the left Node */
            T leftGradient { (2.0/N) * (this->nextNodes[0]->data - this->nextNodes[1]->data) };
            leftGradient = fromGradient(0,0) * leftGradient;
            /* For MSE, rightGradient is the negative of leftGradient. */            
            this->nextNodes[0]->backward(leftGradient);
            this->nextNodes[1]->backward(-leftGradient);
            break;
        }
        default:
            std::cout << "The backward function for " 
                << this->gradientFunction 
                << " is not implemented yet.\n";
            throw std::invalid_argument(
                "The gradient function is not implemented yet."
                " See the last error message above."
            );
    }

}

void Node::backward(double fromGradient)
{
    assert(this->data.size() == 1 && "Backward without parameter should be used "
        "on a node with a scalar data (i.e., 1x1 matrix.)");
    Eigen::MatrixXd dummy(1,1);
    dummy << fromGradient;
    this->backward(dummy);
}

void Node::backward()
{
    /* Backward without parameter can only be used with a 1x1 data. */
    this->backward(1.0);
}


std::ostream& operator<< (std::ostream &out, const std::shared_ptr<Node>& node)
{
    out << node->data;
    return out;    
}

int Node::descendents(int level, bool verbose)
{   
    if (level == 0 && verbose)
    {
        std::cout << "Showing backward graph for node at " << this << '\n';
        std::cout << "Numbers in parentheses shows how many pointers are "
            "referencing the node.\n";
    }
        
        
    if (verbose)
        std::cout << std::string(level * 4, '=') 
            << level << ": " 
            << this->gradientFunction 
            /* minus use count by 1 because 
            this function creates a temporary copy of this. */
            << " (" << shared_from_this().use_count() - 1
            << ", [" << this << ']'
            << ")\n";
    
    if (this->nextNodes.size() == 0){ return 1; }

    for (const std::shared_ptr<Node>& nodePtr: this->nextNodes)
    {
        nodePtr->descendents(level+1, verbose);
    }
    // Calculate number of nodes using DFS
    std::set<std::shared_ptr<Node>> visited {};
    std::vector<std::shared_ptr<Node>> stack {shared_from_this()};
    while (stack.size() > 0)
    {
        std::shared_ptr<Node> curr {stack.back()};
        stack.pop_back();
        if (visited.find(curr) == visited.end())
        {
            visited.insert(curr);

            for (std::shared_ptr<Node> nextNode: curr->nextNodes)
            {
                if (visited.find(nextNode) == visited.end())
                    stack.push_back(nextNode);
            }
        }
    }

    return static_cast<int>(visited.size());
}

int Node::descendents(bool verbose)
{
    return Node::descendents(0, verbose);
}

void Node::visualizeGraph(std::string path)
{
    int maxNode { 1 }; // Maximum number of nodes in a layer.
    int numLayers { 0 };
    // Create row structures
    std::vector<std::vector<NSP>> structure {{shared_from_this()}};
    {
    std::set<NSP> visited {shared_from_this()};
    while (true)
    {
        // Exhaust all children 
        std::vector<NSP> par {structure.back()};
        std::vector<NSP> children {};
        for (NSP p: par)
        {
            auto nextnodes { p->nextNodes };
            std::vector<NSP> uniqueNextNodes {};
            for (NSP nextnode: nextnodes)
            {
                if (visited.find(nextnode) == visited.end())
                {
                    uniqueNextNodes.push_back(nextnode);
                    visited.insert(nextnode);
                }
            }
            children.insert(children.end(), uniqueNextNodes.begin(), uniqueNextNodes.end());
        }
        const int nChildren { static_cast<int>(children.size()) };
        if (nChildren == 0)
            break;
        maxNode = std::max(maxNode, nChildren);
        structure.push_back(children);
    }
    }
    
    numLayers = static_cast<int>(structure.size());

    // Show result
    // for (std::vector<NSP> alayer: structure)
    // {
    //     for (NSP anode: alayer)
    //     {
    //         std::cout << anode -> gradientFunction << ' ';
    //     }
    //     std::cout << '\n';
    // }

    // Plot into SVG
    /* Use rectangle with height and width (h, w) to represent a node.
    Recommend all numbers be even integers. */
    int h {50};
    int w {250};
    int gapH {100}; // Vertical gap between two nodes
    int gapW {50}; // Horizontal gap between two nodes
    int margin {30}; // Margin space of four borders
    int marginT {margin}, marginB {margin}, marginL {margin}, marginR {margin};
    int svgWidth {marginL + marginR + maxNode * (w + gapW) - gapW};
    int svgHeight {marginT + marginB + numLayers * (h + gapH) - gapH};
    // This map record the location (x,y) of each NSP.
    std::unordered_map<NSP, Point> locator {};
    Dimensions dimensions(svgWidth, svgHeight);
    Document graph(path, Layout(dimensions, Layout::TopLeft));

    {
    // Add border
    Polygon border(Stroke(1, Color::Green));
    border << Point(0, 0) << Point(dimensions.width, 0)
        << Point(dimensions.width, dimensions.height) << Point(0, dimensions.height);
    graph << border;

    
    }

    // Add nodes
    for (int level=0; level<numLayers; ++level)
    {
        const int y { marginT + level * (h + gapH) };
        const std::vector<NSP> thisLayer { structure[level] };
        const int thisLayerN { static_cast<int>(thisLayer.size()) };
        const int xBase { marginL + (maxNode - thisLayerN) * (w + gapW) / 2 };
        for (int nodeInd=0; nodeInd<thisLayerN; ++nodeInd)
        {
            NSP thisNode {thisLayer[nodeInd]};
            const int x { xBase + nodeInd * (w+gapW) };
            const int textX { x + w/8 };
            const int textY { y + h/2 };
            std::string dimensionStr {};
            dimensionStr += " (row=" + std::to_string(thisNode->data.rows())
                         + ",col=" + std::to_string(thisNode->data.cols()) + ")";
            graph << Rectangle(Point(x, y), w, h, Color::White, Stroke(3, Color::Black));
            graph << Text(Point(textX, textY), ToString(thisNode->gradientFunction) + dimensionStr,
                Color::Black);
            locator[thisNode] = Point(x,y);
        }
    }

    // Add links by referring to locator
    for (std::unordered_map<NSP, Point>::iterator it=locator.begin(); 
        it!=locator.end(); ++it)
    {
        NSP fromThisNSP {(*it).first};
        Point fromThisPoint {(*it).second};
        for (NSP toThisNSP: fromThisNSP -> nextNodes)
        {
            graph << svgUtility::getLink(fromThisPoint, locator[toThisNSP], w, h);
        }
    }




    graph.save();
}

}
