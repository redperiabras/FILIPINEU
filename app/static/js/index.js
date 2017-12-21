L.mapbox.accessToken = 'pk.eyJ1IjoibGVtYXgiLCJhIjoidnNDV1kzNCJ9.iH26jLhEuimYd6vLOO6v1g';

    var map = L.mapbox.map('map', 'mapbox.outdoors', {
        maxZoom: 20,
        fullscreenControl: true,
        zoomControl: true
    })

    var layers = {
        "Baisc": L.mapbox.tileLayer('mapbox.outdoors').addTo(map),
        "Light": L.mapbox.tileLayer('mapbox.light'),
        "Dark": L.mapbox.tileLayer('mapbox.dark'),
        "Comics": L.mapbox.tileLayer('mapbox.comic'),
        "Pencil": L.mapbox.tileLayer('mapbox.pencil')
    }

    L.control.layers(
        layers,
        null,
        {position: 'topleft'}
    ).addTo(map);

	map.setView({{[14.5519451, 120.9933676]}}, 13);

    markers = [];

    function remove_points(points){
        for (i in points){
            map.removeLayer(points[i])
        }
    }

    function refresh_points(){
        $.ajax({
            type: "POST",
            async: true,
            url: "/map/refresh"
        }).done(function(response){
            remove_points(markers);
            points = response.points;
            for (i in points){
                marker = L.marker(points[i]);
                markers.push(marker);
                marker.addTo(map);
            }
        });
    }