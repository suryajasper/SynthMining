import m from 'mithril';
import './css/main.scss';
import Login from './views/login';
import Main from './views/main';
import ProjectPage from './views/project-page';

function LoginRouter() : m.Component {
  return {
    view(vnode) {
      return m(Login, {signup: false,});
    },
  };
}

function SignupRouter() : m.Component {
  return {
    view(vnode) {
      return m(Login, {signup: true,});
    },
  };
}

m.route(document.body, '/', {
  '/': LoginRouter,
  '/login': LoginRouter,
  '/signup': SignupRouter,
  '/main': Main,
  '/project': ProjectPage,
})