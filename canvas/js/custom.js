// Initialize LRP results
$('#lrp-results').hide();

// Canvas setup
var canvas = new fabric.Canvas('canvas');
canvas.isDrawingMode = true;
canvas.freeDrawingBrush.width = 10;
canvas.freeDrawingBrush.color = "#000000";
canvas.backgroundColor = "#ffffff";
canvas.renderAll();


// Clear button callback
$("#clear-canvas").click(function(){ 
  canvas.clear(); 
  canvas.backgroundColor = "#ffffff";
  canvas.renderAll();
  $("#status").removeClass();
  $('#lrp-results').hide();
  $('#predictionLabel').hide();
  $('#2predictionLabel').hide();
});


// Predict button callback
$("#predict").click(function(){  

  var modelList = document.getElementById("model");
  var model = modelList.options[modelList.selectedIndex].value;

  var lrpType = document.getElementById("lrptype");
  var lrp = lrpType.options[lrpType.selectedIndex].value;

  // var methodType = document.getElementById("method");
  // var method = methodType.options[methodType.selectedIndex].value;
  var method = 3;

  // var methodThreshold = document.getElementById("methodthreshold").value;
  var methodThreshold = 20;

  // var overlapThreshold = document.getElementById("overlapthreshold").value;
  var overlapThreshold = 10;

  // Change status indicator
  $("#status").removeClass().toggleClass("fa fa-spinner fa-spin");

  // Get canvas contents as url
  // var fac = (1.) / 13.; 
  var url = canvas.toDataURL();

  var param = "?mdl"+model+"lrp"+lrp+"mtd"+method+"mth"+methodThreshold+"oth"+overlapThreshold
  // Post url to python script
  var jq = $.post('cgi-bin/mnist.py', url+param)
    .done(function (json) {
      if (json.result) {
        var labels = ['A', 'B', 'C', 'D', 'E']
        $("#status").removeClass().toggleClass("fa fa-check");
        updatelrpresults();
        $('#lrp-results').show();
        console.log(json.error);  
        document.getElementById('predictionLabel').innerHTML = json.indices[0];
        $('#predictionLabel').show();
        document.getElementById('2predictionLabel').innerHTML = json.indices[1];
        $('#2predictionLabel').show();
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