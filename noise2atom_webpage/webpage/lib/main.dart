import 'dart:async';
import 'package:video_player/video_player.dart';
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:url_launcher/url_launcher.dart';

launchURL(String url) async
{
  if (await canLaunch(url))
    await launch(url);
  else
    throw 'Could not launch $url';
}

Widget makeButton( String caption, [String url] )
{
    if (url == null)
        return FlatButton
        (
            color: Colors.blue,
            highlightColor: Colors.blue[700],
            colorBrightness: Brightness.dark,
            splashColor: Colors.grey,
            child: Text(caption),
            shape:RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
            onPressed: () {},
        );
    return FlatButton
    (
        color: Colors.blue[600],
        highlightColor: Colors.blue[900],
        colorBrightness: Brightness.dark,
        splashColor: Colors.grey,
        child: Text(caption),
        shape:RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
        onPressed: () { launchURL(url); },
    );
}
/*
Widget makeButton( String caption )
{
	return FlatButton
    (
        color: Colors.blue,
        highlightColor: Colors.blue[700],
        colorBrightness: Brightness.dark,
        splashColor: Colors.grey,
        child: Text(caption),
        shape:RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
        onPressed: () {},
	);
}
*/

class VideoPlayerScreen extends StatefulWidget
{
    VideoPlayerScreen({Key key, this.caption, this.url}) : super( key:key);
    final String caption;
    final String url;

    @override
    _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen>
{
    VideoPlayerController _controller;
    Future<void> _initializeVideoPlayerFuture;
    @override
    void initState()
    {

        _controller = VideoPlayerController.network( widget.url );
        _initializeVideoPlayerFuture = _controller.initialize();
        _controller.setLooping(true);
        super.initState();
    }

    @override
    void dispose()
    {
        _controller.dispose();
        super.dispose();
    }

    @override
    Widget build(BuildContext context)
    {
        return Column
        (
            children: <Widget>
            [
                FutureBuilder
                (
                    future: _initializeVideoPlayerFuture,
                    builder: (context, snapshot)
                    {
                      if (snapshot.connectionState == ConnectionState.done)
                          return AspectRatio( aspectRatio: _controller.value.aspectRatio, child: VideoPlayer(_controller),);
                      return Center(child: CircularProgressIndicator());
                    },
                ),
                Row
                (
                    children: <Widget>
                    [
                        Spacer(),
                        Text( widget.caption ),
                        Spacer(),
                        Text( 'Click to play/pause ⟿' ),
                        IconButton
                        (
                            onPressed: () { setState(() { if (_controller.value.isPlaying) { _controller.pause(); } else { _controller.play(); } }); },
                            //icon: Icon( Icons.play_arrow ),
                            icon: Icon( _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,),
                            color: Colors.blue,
                            //child: Icon( _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,),
                        ),
                    ],
                ),
            ],
        );
    }
}


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Noise2Atom: Unsupervised Denoising for Scanning Transmission Electron Microscopy Images',

      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(title: 'Noise2Atom: Unsupervised Denoising for Scanning Transmission Electron Microscopy Images'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);
  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {


  @override
  Widget build(BuildContext context) {


    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title, style: TextStyle(fontWeight: FontWeight.bold),),
        centerTitle: true,
        primary: true,
      ),


	body:ListView
	(
		children:
		[
            Container // authors
            (
                height: 56.0, // in logical pixels
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                decoration: BoxDecoration(color: Colors.blue[300]),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Feng Wang'),
                        Spacer(),
                        Text('Trond R Henninen'),
                        Spacer(),
                        Text('Debora Keller'),
                        Spacer(),
                        Text('Rolf Erni'),
                        Spacer(),
                    ],
                ),
            ),
            Container // Address
            (
                height: 56.0, // in logical pixels
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                decoration: BoxDecoration(color: Colors.blue[300]),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Electron Microscopy Center, Swiss Federal Laboratories for Materials Science and Technology, Überland Str. 129, CH-8600 Dübendorf, Switzerland'),
                        Spacer(),
                    ],
                ),
            ),

            Container // demo image
            (
                //!height: 1000.0, // in logical pixels
                //padding: const EdgeInsets.symmetric(horizontal: 20.0),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                //decoration: BoxDecoration(color: Colors.blue[50]),
                child: Image.asset('assets/images/noise2atom_demo.png'),
            ),

            Container // demo image caption
            (
                height: 60.0, // in logical pixels
                //padding: const EdgeInsets.symmetric(horizontal: 8.0),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                //decoration: BoxDecoration(color: Colors.blue[50]),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Noise2Atom is a deep learning model to denoise STEM image series, requiring no signal prior, no noise model estimation, and no paired training images.', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),


            Container
            (
                decoration: BoxDecoration(color: Colors.blue[300]),
                //padding: EdgeInsets.all(30),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                    child: Text('We propose an effective deep learning model to denoise scanning transmission electron microscopy (STEM) image series, named Noise2Atom, to map images from a source domain S to a target domain C, where S is for our noisy experimental dataset, and C is for the desired clear atomic images.  Noise2Atom uses two external networks to apply additional constraints from the domain knowledge.  This model requires no signal prior, no noise model estimation, and no paired training images.  The only assumption is that the inputs are acquired with identical experimental configurations.  To evaluate the restoration performance of our model, as it is impossible to obtain ground truth for our experimental dataset, we propose consecutive structural similarity (CSS) for image quality assessment, based on the fact that the structures remain much the same as the previous frame(s) within small scan intervals.  We demonstrate the superiority of our model by providing evaluation in terms of CSS and visual quality on different experimental datasets.' ),
                ),
            ),

            // Movie
            // https://github.com/fengwang/fengwang.github.io/raw/master/noise2atom/assets/assets/movies/1024x1024_32.tif.mkv
            // https://github.com/fengwang/fengwang.github.io/raw/master/noise2atom/assets/assets/movies/512x512_128.tif.mkv
            // https://github.com/fengwang/fengwang.github.io/raw/master/noise2atom/assets/assets/movies/128x128_1024.tif.mkv


            Container
            (
                //decoration: BoxDecoration(color: Colors.lightBlue[100]),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: 'http://fengwang.github.io/noise2atom/assets/assets/movies/1024x1024_32.tif.mkv',
                        caption: 'Left: experimental images (1024x1024) recorded at 5 fps; Right: denoising results.',
                     ),
                ),
            ),

            Container
            (
                decoration: BoxDecoration(color: Colors.lightBlue[100]),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: 'http://fengwang.github.io/noise2atom/assets/assets/movies/512x512_128.tif.mkv',
                        caption: 'Left: experimental images (512x512) recorded at 15 fps; Right: denoising results.',
                     ),
                ),
            ),

            Container
            (
                //decoration: BoxDecoration(color: Colors.lightBlue[500]),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: 'http://fengwang.github.io/noise2atom/assets/assets/movies/128x128_1024.tif.mkv',
                        caption: 'Left: experimental images (128x128) recorded at 150 fps; Right: denoising results.',
                     ),
                ),
            ),


            Container // buttons
            (
                decoration: BoxDecoration(color: Colors.lightBlue[100]),
                //padding: const EdgeInsets.symmetric(horizontal: 220.0, vertical: 20.0),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        makeButton( 'Paper', 'https://link.springer.com/article/10.1186/s42649-020-00041-8' ),
                        Spacer(),
                        makeButton( 'Code', 'https://github.com/fengwang/Noise2Atom' ),
                        Spacer(),
                        makeButton( 'Demo' ),
                        Spacer(),
                    ],
                ),
            ),

        ],
    ),



        /*
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(

          // horizontal).
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Some Text here',
            ),
          ],
        ),*/
    );
  }
}

