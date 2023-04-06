const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');

/*
 * SplitChunksPlugin is enabled by default and replaced
 * deprecated CommonsChunkPlugin. It automatically identifies modules which
 * should be splitted of chunk by heuristics using module duplication count and
 * module category (i. e. node_modules). And splits the chunks…
 *
 * It is safe to remove "splitChunks" from the generated configuration
 * and was added as an educational example.
 *
 * https://webpack.js.org/plugins/split-chunks-plugin/
 *
 */

/*
 * We've enabled TerserPlugin for you! This minifies your app
 * in order to load faster and run less javascript.
 *
 * https://github.com/webpack-contrib/terser-webpack-plugin
 *
 */

const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  entry: './src/index.jsx',
  mode: 'development',
  plugins: [
    new webpack.ProgressPlugin(),
    new HtmlWebpackPlugin({
      title: 'Synth Mining',
      meta: {
        viewport: 'width=device-width, initial-scale=1, shrink-to-fit=no, viewport-fit=cover',
        'application-name': 'SynthMining',
        description: 'A platform that solves some of the biggest roadblocks in image-based data science by combining collaborative outsourced datamining with synthetic image generation',
      },
    }),
    new webpack.DefinePlugin({
      'process.env.ENVIRONMENT': "'BROWSER'"
    }),
    new webpack.ProvidePlugin({
      process: 'process/browser',
    }),
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
    }),
  ],

  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
    fallback: {
      "fs": false,
      "tls": false,
      "net": false,
      "path": false,
      "http": false,
      "https": false,
      "stream": false,
      "assert": false,
    }
  },

  module: {
    rules: [
      {
        test: /\.(ts|tsx|js|jsx)$/,
        include: [path.resolve(__dirname, 'src')],
        use: [
          {
            loader: 'babel-loader',   
            options: {
              presets: [
                '@babel/preset-typescript',
                '@babel/preset-react',
              ]
            }         
          },
        ],
      }, 
      {
        test: /.css$/,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            options: {
              modules: {
                localIdentName: '[local]',
              },
              sourceMap: true,
            },
          },
          'sass-loader',
        ],
      }, 
      {
        test: /\.(png|jpg|gif)/,
        exclude: /node_modules/,
        use: [
          {
            loader: require.resolve('file-loader'),
            options: {
              name: 'static/images/[contenthash].[ext]',
              esModule: false,
            }
          },
        ]
      }
    ],
  },

  output: {
    filename: 'static/js/build.[contenthash].js',
    path: path.join(__dirname, '/dist'),
    chunkFilename: 'static/js/[name].[contenthash].js',
    publicPath: '/',
  },

  optimization: {
    minimizer: [new TerserPlugin()],

    splitChunks: {
      cacheGroups: {
        vendors: {
          priority: -10,
          test: /[\\/]node_modules[\\/]/,
        },
      },

      chunks: 'async',
      minChunks: 1,
      minSize: 30000,
      name: false,
    },
  },
};
