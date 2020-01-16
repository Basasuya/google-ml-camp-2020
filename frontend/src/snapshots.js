import React, { Component } from "react"
import * as d3 from "d3"
import d3Lasso from './d3Lasso'
// import 'd3-scale-chromatic'
// import './d3-lasso'
class Snapshots extends Component {
    constructor(props) {
        super(props)
    }


    handleLasso (chooseList = []) {
        if (this.props.onLasso) {
            this.props.onLasso(chooseList)
        }
    }

    componentWillReceiveProps(props) {
        // console.log(props, this.props)
        if(this.props.snapshots == props.snapshots) {
            let container = d3.select("#snapshot").selectAll("g.container")
            container.selectAll(".new").remove()
            return
        }

        console.log('snapshots will receive props')
        const self = this
        const snapshots = props.snapshots
        const svg = d3.select("#snapshot")
        const padding = 20
        const clientWidth = svg.node().parentNode.clientWidth;
        const clientHeight = document.body.clientHeight;
        const width = Math.min(clientWidth, clientHeight)
        svg.attr("width", width).attr("height", width)
        // svg.call(
        //     d3.zoom()
        //         .scaleExtent([.1, 3])
        //         .on("zoom", function() {  container.attr("transform", d3.event.transform);})
        // );

        // console.log(snapshots)
        const height = width;

        this.props.onSetHeight(height);

        const max = {}
        const min = {}
        max.x = d3.max(snapshots, snpst => snpst.positionX)
        max.y = d3.max(snapshots, snpst => snpst.positionY)
        min.x = d3.min(snapshots, snpst => snpst.positionX)
        min.y = d3.min(snapshots, snpst => snpst.positionY)
        
        let maxLabel = d3.max(snapshots, snpst => snpst.label)

        console.log(maxLabel)
        // print("hhh", max, min)
        const xScale = d3
            .scaleLinear()
            .domain([min.x, max.x])
            .range([padding, width - padding])
        const yScale = d3
            .scaleLinear()
            .domain([min.y, max.y])
            .range([padding, width - padding])
        
        // var rect = svg.append("rect")
        //     .attr("class", "overlay")
        //     .attr("width", width)
        //     .attr("height", height)
        //     .attr("fill", 'white')
        // rect.on('click', unfocus)

        // function unfocus() {
        //     // let circles = container.selectAll(".snapdot")
        //     container.selectAll(".snapdot").remove()
        //     lasso.items().style("fill", d => colors(d.du)).style("fill", 3)
        // }
        
        var lasso_start = function() {
            lasso.items()
            .attr("r",3) // reset size
            .style("fill",null) // clear all of the fills
            .classed('not_possible', true).classed('selected', false); // style as not possible
        };
        
        var lasso_draw = function() {
        // Style the possible dots
            lasso.items().filter(function(d) {return d.possible===true})
            .classed('not_possible', false).classed('possible', true)
            
            // Style the not possible dot
            lasso.items().filter(function(d) {return d.possible===false})
            .classed('not_possible', true).classed('possible', false)
        };
        
        var lasso_end = function() {
        // Reset the color of all dots
            lasso.items()
               .style("fill", d => colors(d.label, maxLabel));
        
        // Style the selected dots
            let filterResult = []
            lasso.items().filter(function(d) { 
                if(d.selected ===  true) {
                    // console.log(d)
                    filterResult.push(d.key); 
                }
                return d.selected===true})
            .classed('not_possible', false).classed('possible', false)
            .attr("r",5).style("fill", '#ffff66')
        
        // Reset the style of the not selected dots
            lasso.items().filter(function(d) {return d.selected===false})
            .classed('not_possible', false).classed('possible', false)
            .attr("r",3);

            // console.log(filterResult)
            self.handleLasso(filterResult)
        };
        
        // Create the area where the lasso event can be triggered
        var lasso_area = svg.append("rect")
                            .attr("width",width)
                            .attr("height",height)
                            .style("opacity",0);
        
        // Define the lasso
        var lasso = d3Lasso()
            .closePathDistance(75) // max distance for the lasso loop to be closed
            .closePathSelect(true) // can items be selected by closing the path?
            .hoverSelect(true) // can items by selected by hovering over them?
            .area(lasso_area) // area where the lasso can be started
            .on("start",lasso_start) // lasso start function
            .on("draw",lasso_draw) // lasso draw function
            .on("end",lasso_end); // lasso end function
        
        // Init the lasso on the svg:g that contains the dots
        svg.call(lasso);
        
        // svg.call(handleLasso);
        // lasso.items(circles);
        
        // var a = d3.rgb(245, 255, 245)
        // var b = d3.rgb(0, 179, 0)
        // var compute = d3.interpolate(a,b);
        // var maxDu = d3.max(snapshots, n => n.du)
        // var minDu = d3.min(snapshots, n => n.du)
        // console.log('du', minDu, maxDu)
        // console.log(maxDu, minDu)
        let blueArray = ['#d0e5fb','#b8d9f9','#a1ccf7','#89bff6','#71b2f4','#5aa6f2','#4299f0','#2a8cee','#137fec','#1173d5','#0f66bd','#0d59a5','#0b4c8e']
        let redArray = ['#ffe5e5','#ffcccc','#ffb3b3','#ff9999','#ff8080','#ff6666','#ff4d4d','#ff3333','#ff1a1a','#ff0000','#e60000','#cc0000','#b30000']
        // blueArray.reverse()
        // let blueColors = d3.scaleQuantize()
        // .domain([minDu, 0])
        // .range(blueArray)

        // let redColors = d3.scaleQuantize()
        // .domain([0, maxDu])
        // .range(redArray)

        let colors = (x, maxLabel) => {
            return d3.interpolateCubehelixDefault(x / maxLabel)
            // d3.interpolateHsl("red", "blue")(x / maxLabel)
            // return blueArray[0]
        }


        // .range(["#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4", "#E6F598","#FFFFBF", "#FEE08B"]);
        // , "#E6F598", 
        // "#FFFFBF", "#FEE08B", "#FDAE61", "#F46D43", "#D53E4F", "#9E0142"
        
                  
        let container = svg.append("g").classed("container", true)
        // container.selectAll(".new").remove()
        const points = container.selectAll("circle").data(snapshots)
        points.exit().remove()
        points
            .enter()
            .append("circle")
            .classed('snapdot', true)
            .attr("cx", d => xScale(d.positionX))
            .attr("cy", d => yScale(d.positionY))
            .attr("r", 3)
            .style("fill", d => colors(d.label, maxLabel))
            .on("click", (d, i) => {
                let result = []
                result.push(d.key)
                self.handleLasso(result)
            })
        
        //     lasso.area(svg.selectAll(".lassoable"));
        lasso.items(container.selectAll('circle'));
    }
    render() {
        return <svg id="snapshot" />
    }
}

export default Snapshots
