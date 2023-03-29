import React from 'react';
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import './css/main.scss';

import Login from './views/login';
import Main from './views/main';

const HelloWorld = () => <div>{"Fuck"}</div>

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/">
          <Route index element={<Login />} />
          <Route path="login" element={<Login />} />
          <Route path="signup" element={<Login signup />} />
          <Route path='main' element={<Main />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

const root = ReactDOM.createRoot(document.body);

root.render(<App />);