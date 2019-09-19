// Initialize LRP results
$('#lrp-results').hide();

// Canvas setup
var canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 15;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();


// Clear button callback
$("#clear-canvas").click(function(){ 
  canvas.clear(); 
  canvas.backgroundColor = "#ffffff";
  canvas.renderAll();
  updateChart(zeros);
  $("#status").removeClass();
  $('#lrp-results').hide();
});


// Predict button callback
$("#predict").click(function(){  

  // Change status indicator
  $("#status").removeClass().toggleClass("fa fa-spinner fa-spin");

  // Get canvas contents as url
  var fac = (1.) / 13.; 
  var url = canvas.toDataURLWithMultiplier('png', fac);

  // Post url to python script
  var jq = $.post('cgi-bin/mnist.py', url)
    .done(function (json) {
      if (json.result) {
        var labels = ['A', 'B', 'C', 'D', 'E']
        $("#status").removeClass().toggleClass("fa fa-check");
        $('#svg-chart').show();
        updateChart(json.data, labels);
        updatelrpresults();
        $('#lrp-results').show();
      } else {
         $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
         console.log('Script Error: ' + json.error)
      }
    })
    .fail(function (xhr, textStatus, error) {
      $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
      console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
    }
  );

});

// Show LRP results
function updatelrpresults() {
  $("#lrpimg").prop("src", "/images/lrpresult.png?" + new Date().valueOf());
}  

// Iniitialize d3 bar chart
$('#svg-chart').show();
var zeros = [0,0,0,0,0]
var labels = ['A', 'B', 'C', 'D', 'E']

var margin = {top: 20, right: 20, bottom: 20, left: 20},
    width = 360 - margin.left - margin.right,
    height = 180 - margin.top - margin.bottom;

var x = d3.scale.linear()
    .range([0, width]);

var y = d3.scale.ordinal()
    .rangeRoundBands([0, height], 0.1);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickSize(0)
    .tickPadding(6);

var svg = d3.select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  x.domain([-5, 5]);
  y.domain(labels);

  svg.selectAll(".bar")
      .data(zeros)
    .enter().append("rect")
      .attr("class", function(d) { return "bar bar--" + (d < 0 ? "negative" : "positive"); })
      .attr("x", function(d) { return x(Math.min(0, d)); })
      .attr("y", function(d) { return y(d); })
      .attr("width", function(d) { return (x(d) - x(0)); })
      .attr("height", y.rangeBand());

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .attr("transform", "translate(" + x(0) + ",0)")
      .call(yAxis);

// Update chart data
function updateChart(data, labels) {
  full = {"data": [data], "labels": [labels]}
  svg.selectAll(".bar")
      .data(full)
    .enter().append("rect")
      .attr("class", function(d) { return "bar bar--" + (d < 0 ? "negative" : "positive"); })
      .attr("x", function(d) { return x(Math.min(0, d.data)); })
      .attr("y", function(d) { return y(d.labels); })
      .attr("width", function(d) { return Math.abs(x(d.data) - x(0)); })
      .attr("height", y.rangeBand());
}
